#!/usr/bin/env python3
"""
处理电影文件的脚本

支持两种预处理任务（可并行）：
1. WhisperX：语音转文字、词级对齐、说话人分离 → .json + .srt
2. TransNet：镜头检测与切片 → shots/ + shots_list.txt

采用生产者-消费者模式：
- 生产者：扫描 Movies 目录，将任务放入队列
- 消费者：WhisperX 与 TransNet 各自有独立 worker 池，可多卡并行

支持两种模式：
- 批量模式 (--batch)：扫描 Datasets/Movies，结果保存到 Datasets/AVAGen/{电影名}/
- 单文件模式：指定单个文件处理
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from queue import Empty
from typing import Optional

# 确保能导入 tools 下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 默认路径（相对于项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MOVIES_DIR = os.path.join(PROJECT_ROOT, "Datasets", "Movies")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Datasets", "AVAGen")

# 任务类型
TASK_WHISPERX = "whisperx"
TASK_TRANSNET = "transnet"


def collect_mp4_files(movies_dir: str) -> list[tuple[str, str]]:
    """递归收集 movies_dir 下所有 mp4 文件，返回 (path, movie_name) 列表"""
    videos = []
    for root, _, files in os.walk(movies_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                path = os.path.join(root, f)
                name = os.path.splitext(f)[0]
                videos.append((path, name))
    return sorted(videos, key=lambda x: x[0])


# ============== WhisperX Worker ==============
def _whisperx_worker(
    worker_id: int,
    gpu_id: int,
    task_queue: mp.Queue,
    output_dir: str,
    whisper_model: str,
    batch_size: int,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    no_srt: bool,
    hf_token: str,
    done_sentinel: object,
) -> None:
    """WhisperX 消费者进程"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from tools.whisperx_pipeline import MovieTranscriber

    transcriber = MovieTranscriber(hf_token=hf_token, device="cuda")

    while True:
        try:
            item = task_queue.get(timeout=1)
        except Empty:
            continue
        if item is done_sentinel:
            break

        video_path, movie_name = item
        out_folder = os.path.join(output_dir, movie_name)
        out_prefix = os.path.join(out_folder, movie_name)

        try:
            result = transcriber.process(
                input_path=video_path,
                whisper_model=whisper_model,
                batch_size=batch_size,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            os.makedirs(out_folder, exist_ok=True)
            json_path = f"{out_prefix}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  [WhisperX W{worker_id}] {movie_name}: JSON 已保存")

            if not no_srt:
                transcriber.save_srt(result, f"{out_prefix}.srt")
                print(f"  [WhisperX W{worker_id}] {movie_name}: SRT 已保存")
        except Exception as e:
            print(f"  [WhisperX W{worker_id}] {movie_name}: 错误 - {e}")
            raise


# ============== TransNet Worker ==============
def _transnet_worker(
    worker_id: int,
    gpu_id: int,
    task_queue: mp.Queue,
    output_dir: str,
    transnet_threshold: float,
    transnet_min_frames: int,
    transnet_save_mp4: bool,
    done_sentinel: object,
) -> None:
    """TransNet 消费者进程"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from tools.transnet_pipeline import TransNetV2Pipeline

    pipeline = TransNetV2Pipeline(device="cuda")

    while True:
        try:
            item = task_queue.get(timeout=1)
        except Empty:
            continue
        if item is done_sentinel:
            break

        video_path, movie_name = item
        out_folder = os.path.join(output_dir, movie_name)

        try:
            pipeline.read_video(video_path)
            shot_count = pipeline.shot_detect(
                output_dir=out_folder,
                threshold=transnet_threshold,
                min_frames=transnet_min_frames,
                save_mp4=transnet_save_mp4,
            )
            print(f"  [TransNet W{worker_id}] {movie_name}: 检测到 {shot_count} 个镜头")
        except Exception as e:
            print(f"  [TransNet W{worker_id}] {movie_name}: 错误 - {e}")
            raise


# ============== 生产者 ==============
def _producer(
    whisperx_queue: mp.Queue,
    transnet_queue: mp.Queue,
    videos: list[tuple[str, str]],
    enable_whisperx: bool,
    enable_transnet: bool,
    whisperx_workers: int,
    transnet_workers: int,
    done_sentinel: object,
) -> None:
    """生产者：将任务放入两个队列"""
    for video_path, movie_name in videos:
        if enable_whisperx:
            whisperx_queue.put((video_path, movie_name))
        if enable_transnet:
            transnet_queue.put((video_path, movie_name))

    # 发送结束信号（每个 worker 一个）
    if enable_whisperx:
        for _ in range(whisperx_workers):
            whisperx_queue.put(done_sentinel)
    if enable_transnet:
        for _ in range(transnet_workers):
            transnet_queue.put(done_sentinel)


def run_batch_parallel(args) -> None:
    """批量模式 + 生产者消费者并行"""
    movies_dir = args.movies_dir or DEFAULT_MOVIES_DIR
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR

    if not os.path.isdir(movies_dir):
        print(f"错误: Movies 目录不存在: {movies_dir}")
        sys.exit(1)

    videos = collect_mp4_files(movies_dir)
    if not videos:
        print(f"未在 {movies_dir} 下找到任何 mp4 文件")
        return

    enable_whisperx = args.whisperx_workers > 0
    enable_transnet = args.transnet_workers > 0
    if not enable_whisperx and not enable_transnet:
        print("错误: 至少启用一种任务 (--whisperx-workers > 0 或 --transnet-workers > 0)")
        sys.exit(1)

    if enable_whisperx and not args.hf_token:
        print("错误: WhisperX 需要 export HUGGINGFACE_TOKEN")
        sys.exit(1)

    print(f"找到 {len(videos)} 个 mp4 文件")
    print(f"输出目录: {output_dir}")
    if enable_whisperx:
        print(f"WhisperX: {args.whisperx_workers} workers, GPUs {args.whisperx_gpus}")
    if enable_transnet:
        print(f"TransNet: {args.transnet_workers} workers, GPUs {args.transnet_gpus}")
    print("-" * 50)

    mp.set_start_method("spawn", force=True)

    done_sentinel = object()

    whisperx_queue = mp.Queue() if enable_whisperx else None
    transnet_queue = mp.Queue() if enable_transnet else None

    processes = []

    # 启动 WhisperX workers
    if enable_whisperx:
        for i in range(args.whisperx_workers):
            gpu_id = args.whisperx_gpus[i % len(args.whisperx_gpus)]
            p = mp.Process(
                target=_whisperx_worker,
                args=(
                    i,
                    gpu_id,
                    whisperx_queue,
                    output_dir,
                    args.whisper_model,
                    args.batch_size,
                    args.min_speakers,
                    args.max_speakers,
                    args.no_srt,
                    args.hf_token,
                    done_sentinel,
                ),
            )
            p.start()
            processes.append(p)

    # 启动 TransNet workers
    if enable_transnet:
        for i in range(args.transnet_workers):
            gpu_id = args.transnet_gpus[i % len(args.transnet_gpus)]
            p = mp.Process(
                target=_transnet_worker,
                args=(
                    i,
                    gpu_id,
                    transnet_queue,
                    output_dir,
                    args.transnet_threshold,
                    args.transnet_min_frames,
                    args.transnet_save_mp4,
                    done_sentinel,
                ),
            )
            p.start()
            processes.append(p)

    # 启动生产者
    producer = mp.Process(
        target=_producer,
        args=(
            whisperx_queue or mp.Queue(),
            transnet_queue or mp.Queue(),
            videos,
            enable_whisperx,
            enable_transnet,
            args.whisperx_workers,
            args.transnet_workers,
            done_sentinel,
        ),
    )
    producer.start()
    processes.append(producer)

    for p in processes:
        p.join()
        if p.exitcode != 0 and p != producer:
            sys.exit(1)

    print(f"\n全部完成，共处理 {len(videos)} 个电影。")


def run_batch_sequential(args) -> None:
    """批量模式：顺序执行（无并行）"""
    movies_dir = args.movies_dir or DEFAULT_MOVIES_DIR
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR

    if not os.path.isdir(movies_dir):
        print(f"错误: Movies 目录不存在: {movies_dir}")
        sys.exit(1)

    videos = collect_mp4_files(movies_dir)
    if not videos:
        print(f"未在 {movies_dir} 下找到任何 mp4 文件")
        return

    print(f"找到 {len(videos)} 个 mp4 文件")
    print(f"输出目录: {output_dir}")
    print("-" * 50)

    from tools.whisperx_pipeline import MovieTranscriber
    from tools.transnet_pipeline import TransNetV2Pipeline

    transcriber = MovieTranscriber(hf_token=args.hf_token)
    transnet = TransNetV2Pipeline()

    for i, (video_path, movie_name) in enumerate(videos, 1):
        out_folder = os.path.join(output_dir, movie_name)
        out_prefix = os.path.join(out_folder, movie_name)
        os.makedirs(out_folder, exist_ok=True)

        print(f"\n[{i}/{len(videos)}] 处理: {video_path}")

        try:
            # WhisperX
            result = transcriber.process(
                input_path=video_path,
                whisper_model=args.whisper_model,
                batch_size=args.batch_size,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )
            with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            if not args.no_srt:
                transcriber.save_srt(result, f"{out_prefix}.srt")
            print(f"  WhisperX 完成")

            # TransNet
            transnet.read_video(video_path)
            shot_count = transnet.shot_detect(
                output_dir=out_folder,
                threshold=args.transnet_threshold,
                min_frames=args.transnet_min_frames,
                save_mp4=args.transnet_save_mp4,
            )
            print(f"  TransNet 完成: {shot_count} 个镜头")
        except Exception as e:
            print(f"  错误: {e}")
            if not args.continue_on_error:
                raise

    print(f"\n全部完成，共处理 {len(videos)} 个电影。")


def run_single(args) -> None:
    """单文件模式"""
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)

    from tools.whisperx_pipeline import MovieTranscriber
    from tools.transnet_pipeline import TransNetV2Pipeline

    movie_name = os.path.splitext(os.path.basename(args.input))[0]
    out_prefix = args.output or os.path.join(os.path.dirname(args.input), movie_name)
    out_folder = os.path.dirname(out_prefix) or "."
    os.makedirs(out_folder, exist_ok=True)

    print(f"输入: {args.input}")
    print(f"输出前缀: {out_prefix}")
    print("-" * 50)

    # WhisperX
    transcriber = MovieTranscriber(hf_token=args.hf_token)
    result = transcriber.process(
        input_path=args.input,
        whisper_model=args.whisper_model,
        batch_size=args.batch_size,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    if not args.no_srt:
        transcriber.save_srt(result, f"{out_prefix}.srt")
    print("WhisperX 完成")

    # TransNet
    transnet = TransNetV2Pipeline()
    transnet.read_video(args.input)
    shot_count = transnet.shot_detect(
        output_dir=out_folder,
        threshold=args.transnet_threshold,
        min_frames=args.transnet_min_frames,
        save_mp4=args.transnet_save_mp4,
    )
    print(f"TransNet 完成: {shot_count} 个镜头")
    print("处理完成。")


def main():
    global args
    parser = argparse.ArgumentParser(
        description="处理电影文件：WhisperX 转录 + TransNet 镜头切片"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量模式：扫描 Datasets/Movies，结果保存到 Datasets/AVAGen/{电影名}/",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="批量模式下启用生产者-消费者并行（需配合 --whisperx-workers/--transnet-workers）",
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="单文件模式：电影/视频文件路径",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="单文件模式：输出路径前缀",
    )
    parser.add_argument(
        "--movies-dir",
        type=str,
        default=None,
        help="批量模式：Movies 目录 (默认: Datasets/Movies)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="批量模式：输出根目录 (默认: Datasets/AVAGen)",
    )
    # WhisperX
    parser.add_argument(
        "--whisperx-workers",
        type=int,
        default=0,
        help="WhisperX worker 数量，0=仅顺序模式时启用",
    )
    parser.add_argument(
        "--whisperx-gpus",
        type=str,
        default="0",
        help="WhisperX 使用的 GPU ID 列表，逗号分隔，如 0,1,2",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Whisper 推理批次大小",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3",
        help="Whisper 模型名称",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="说话人数量下限",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="说话人数量上限",
    )
    parser.add_argument(
        "--no-srt",
        action="store_true",
        help="不生成 SRT 字幕",
    )
    # TransNet
    parser.add_argument(
        "--transnet-workers",
        type=int,
        default=0,
        help="TransNet worker 数量，0=仅顺序模式时启用",
    )
    parser.add_argument(
        "--transnet-gpus",
        type=str,
        default="0",
        help="TransNet 使用的 GPU ID 列表，逗号分隔",
    )
    parser.add_argument(
        "--transnet-threshold",
        type=float,
        default=0.2,
        help="TransNet 镜头检测阈值",
    )
    parser.add_argument(
        "--transnet-min-frames",
        type=int,
        default=0,
        help="TransNet 最小镜头帧数",
    )
    parser.add_argument(
        "--no-transnet-mp4",
        action="store_true",
        help="TransNet 不切分保存 MP4，仅生成 shots_list.txt",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="批量模式：某个文件失败时继续",
    )

    args = parser.parse_args()

    args.hf_token = os.getenv("HUGGINGFACE_TOKEN")
    args.transnet_save_mp4 = not args.no_transnet_mp4
    args.whisperx_gpus = [int(x.strip()) for x in args.whisperx_gpus.split(",")]
    args.transnet_gpus = [int(x.strip()) for x in args.transnet_gpus.split(",")]

    use_parallel = args.batch and args.parallel and (
        args.whisperx_workers > 0 or args.transnet_workers > 0
    )

    if use_parallel and args.hf_token is None and args.whisperx_workers > 0:
        print("错误: 并行模式下启用 WhisperX 需 export HUGGINGFACE_TOKEN")
        sys.exit(1)

    if args.batch:
        if use_parallel:
            run_batch_parallel(args)
        else:
            if args.hf_token is None:
                print("错误: 请先 export HUGGINGFACE_TOKEN=xxx")
                sys.exit(1)
            run_batch_sequential(args)
    else:
        if not args.input:
            parser.error("单文件模式需要指定 input，或使用 --batch 进入批量模式")
        if args.hf_token is None:
            print("错误: 请先 export HUGGINGFACE_TOKEN=xxx")
            sys.exit(1)
        run_single(args)


if __name__ == "__main__":
    main()
