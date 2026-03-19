"""
批量处理 Datasets/Movies 下的 MP4，输出到 Datasets/AVAGen/{电影名}/
支持多卡并行：每张 GPU 一个 worker 进程。
"""
import argparse
import json
import logging
import multiprocessing
import os
import sys
from queue import Empty

import torch
from tqdm import tqdm

# 抑制 WhisperX 等库的中间日志
for _name in ("tools.whisperx_pipeline", "whisperx", "whisperx.asr", "whisperx.vads", "pyannote", "lightning"):
    logging.getLogger(_name).setLevel(logging.WARNING)

from tools.ocr_pipeline import EasyOCRCreditDetectorPipeline
from tools.person_detection_pipeline import PersonDetectionPipeline
from tools.transnet_pipeline import TransNetV2Pipeline
from tools.whisperx_pipeline import MovieTranscriber


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MOVIES_DIR = os.path.join(PROJECT_ROOT, "Datasets", "Movies")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Datasets", "AVAGen")


def collect_mp4_files(movies_dir: str) -> list[tuple[str, str]]:
    """递归收集目录下所有 mp4，返回 (视频路径, 电影名)"""
    videos: list[tuple[str, str]] = []
    for root, _, files in os.walk(movies_dir):
        for filename in files:
            if filename.lower().endswith(".mp4"):
                video_path = os.path.join(root, filename)
                movie_name = os.path.splitext(filename)[0]
                videos.append((video_path, movie_name))
    return sorted(videos, key=lambda item: item[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量转场检测：Movies/*.mp4 -> AVAGen/{电影名}/"
    )
    parser.add_argument(
        "--movies-dir",
        type=str,
        default=DEFAULT_MOVIES_DIR,
        help=f"输入电影目录（默认：{DEFAULT_MOVIES_DIR}）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出根目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Models", "TransNetV2"),
        help="TransNetV2 模型目录（不存在时自动从 HuggingFace 下载）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='单卡模式下的推理设备（多卡时由 --gpus 指定）',
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default='all',
        help='多卡并行时使用的 GPU ID，逗号分隔如 "0,1,2,3"，或 "all" 使用全部可用 GPU。不指定则单卡运行',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="镜头检测阈值（越小越敏感）",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=0,
        help="最小镜头帧数",
    )
    parser.add_argument(
        "--segment-frames",
        type=int,
        default=1000,
        help="每批处理帧数",
    )
    parser.add_argument(
        "--no-save-mp4",
        action="store_true",
        help="不保存切分后的镜头 MP4（仅输出 shots_list.txt）",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="某个电影失败时继续处理下一个",
    )
    parser.add_argument(
        "--no-ocr-check",
        action="store_true",
        help="不进行片头片尾 OCR 检测（仅当保存 MP4 时有效）",
    )
    parser.add_argument(
        "--no-person-check",
        action="store_true",
        help="不进行人物检测（仅当保存 MP4 时有效）",
    )
    parser.add_argument(
        "--no-whisperx",
        action="store_true",
        help="不进行 WhisperX 字幕转录（仅当保存 MP4 时有效，结果保存到 subtitles/）",
    )
    return parser.parse_args()


def _load_existing_clips_check(jsonl_path: str) -> dict[str, dict]:
    """加载已有的 clips_check.jsonl，返回 {filename: record}"""
    if not os.path.isfile(jsonl_path):
        return {}
    cached: dict[str, dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                fn = record.get("file")
                if fn:
                    cached[fn] = record
            except json.JSONDecodeError:
                continue
    return cached


def _record_is_complete(record: dict, no_ocr_check: bool, no_person_check: bool) -> bool:
    """判断已有记录是否完整，可跳过重新检测"""
    if not no_ocr_check:
        if record.get("has_text") is None and "ocr_error" not in record:
            return False
    if not no_person_check:
        if record.get("has_person") is None and "person_error" not in record:
            return False
    return True


def check_credits_and_persons_for_movie(
    movie_output_dir: str,
    movie_name: str,
    ocr_pipeline: EasyOCRCreditDetectorPipeline | None,
    person_pipeline: PersonDetectionPipeline | None,
    no_ocr_check: bool,
    no_person_check: bool,
) -> None:
    """
    对电影输出目录下 shots/ 中的每个片段进行 OCR 和人物检测，
    保存为 JSONL：每行一个 clip 的检查结果。
    已存在于 clips_check.jsonl 且结果完整的片段会跳过。
    """
    shots_dir = os.path.join(movie_output_dir, "shots")
    if not os.path.isdir(shots_dir):
        return

    mp4_files = sorted(
        f for f in os.listdir(shots_dir) if f.lower().endswith(".mp4")
    )
    if not mp4_files:
        return

    jsonl_path = os.path.join(movie_output_dir, "clips_check.jsonl")
    cached = _load_existing_clips_check(jsonl_path)

    results: list[dict] = []
    to_process = [
        f for f in mp4_files
        if not _record_is_complete(cached.get(f, {}), no_ocr_check, no_person_check)
    ]
    skipped = len(mp4_files) - len(to_process)

    for filename in tqdm(mp4_files, desc="OCR + 人物检测"):
        if filename in cached and _record_is_complete(cached[filename], no_ocr_check, no_person_check):
            results.append(cached[filename])
            continue

        video_path = os.path.join(shots_dir, filename)
        record: dict = {"file": filename}

        if not no_ocr_check and ocr_pipeline:
            try:
                is_credit, _ = ocr_pipeline.process_clip(video_path)
                record["has_text"] = is_credit
            except Exception as exc:
                record["has_text"] = None
                record["ocr_error"] = str(exc)
        else:
            record["has_text"] = None

        if not no_person_check and person_pipeline:
            try:
                has_person_result, _ = person_pipeline.process_clip(video_path)
                record["has_person"] = has_person_result
            except Exception as exc:
                record["has_person"] = None
                record["person_error"] = str(exc)
        else:
            record["has_person"] = None

        results.append(record)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    credits_count = sum(1 for r in results if r.get("has_text") is True)
    persons_count = sum(1 for r in results if r.get("has_person") is True)
    msg = f"检测完成：{credits_count} 个有文字，{persons_count} 个有人物"
    if skipped:
        msg += f"，跳过 {skipped} 个已处理"
    print(msg)
    print(f"结果已保存：{jsonl_path}")


def run_whisperx_for_movie(
    movie_output_dir: str,
    transcriber: MovieTranscriber,
) -> None:
    """
    对电影输出目录下 shots/ 中的每个片段运行 WhisperX 转录，
    将 JSON 和 SRT 保存到 subtitles/ 文件夹。
    """
    shots_dir = os.path.join(movie_output_dir, "shots")
    subtitles_dir = os.path.join(movie_output_dir, "subtitles")
    if not os.path.isdir(shots_dir):
        return

    mp4_files = sorted(
        f for f in os.listdir(shots_dir) if f.lower().endswith(".mp4")
    )
    if not mp4_files:
        return

    os.makedirs(subtitles_dir, exist_ok=True)

    for filename in tqdm(mp4_files, desc="WhisperX 转录"):
        video_path = os.path.join(shots_dir, filename)
        base_name = os.path.splitext(filename)[0]
        out_prefix = os.path.join(subtitles_dir, f"{base_name}_whisperx")

        try:
            result = transcriber.process(
                input_path=video_path,
                batch_size=16,
                language="en",
            )
            json_path = f"{out_prefix}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            transcriber.save_srt(result, f"{out_prefix}.srt")
        except Exception as exc:
            print(f"WhisperX 失败 {filename}: {exc}")

    print(f"WhisperX 字幕已保存至：{subtitles_dir}")


def _process_one_movie(
    video_path: str,
    movie_name: str,
    output_dir: str,
    model_path: str,
    device: str,
    threshold: float,
    min_frames: int,
    segment_frames: int,
    save_mp4: bool,
    no_ocr_check: bool,
    no_person_check: bool,
    no_whisperx: bool,
) -> tuple[str, bool, str | int]:
    """
    处理单个电影，返回 (movie_name, success, shot_count_or_error)。
    在 worker 进程内调用，使用该进程的 CUDA_VISIBLE_DEVICES。
    """
    try:
        pipeline = TransNetV2Pipeline(model_path=model_path, device=device)
        pipeline.read_video(video_path)
        movie_output_dir = os.path.join(output_dir, movie_name)
        os.makedirs(movie_output_dir, exist_ok=True)

        shot_count = pipeline.shot_detect(
            output_dir=movie_output_dir,
            threshold=threshold,
            min_frames=min_frames,
            segment_frames=segment_frames,
            save_mp4=save_mp4,
        )

        if save_mp4 and (not no_ocr_check or not no_person_check):
            ocr_pipeline = None
            person_pipeline = None
            if not no_ocr_check:
                ocr_pipeline = EasyOCRCreditDetectorPipeline(
                    sample_interval_sec=1.0, min_text_boxes=1
                )
            if not no_person_check:
                person_pipeline = PersonDetectionPipeline(
                    sample_interval_sec=1.0,
                    min_person_frames=1,
                    confidence_threshold=0.5,
                    device=device,
                )
            check_credits_and_persons_for_movie(
                movie_output_dir,
                movie_name,
                ocr_pipeline,
                person_pipeline,
                no_ocr_check,
                no_person_check,
            )

        if save_mp4 and not no_whisperx:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if hf_token:
                # WhisperX/CTranslate2 仅支持 "cuda" 格式，不支持 "cuda:0"
                # 多卡 worker 已通过 CUDA_VISIBLE_DEVICES 隔离，用 "cuda" 即可
                wx_device = "cuda" if device.startswith("cuda") else device
                transcriber = MovieTranscriber(hf_token=hf_token, device=wx_device)
                run_whisperx_for_movie(movie_output_dir, transcriber)
            else:
                print(f"跳过 WhisperX：未设置 HUGGINGFACE_TOKEN")

        return (movie_name, True, shot_count)
    except Exception as exc:
        return (movie_name, False, str(exc))


def _worker_process(
    gpu_id: int,
    work_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    worker_args: dict,
) -> None:
    """
    Worker 进程：每个进程绑定一张 GPU，从队列取任务并处理。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"  # 本进程只可见一张卡，即 cuda:0

    while True:
        try:
            task = work_queue.get(timeout=1)
        except Empty:
            continue
        if task is None:
            break
        video_path, movie_name = task
        result = _process_one_movie(
            video_path=video_path,
            movie_name=movie_name,
            device=device,
            **worker_args,
        )
        result_queue.put(result)


def _run_single_gpu(args, videos: list) -> int:
    """单卡串行处理。"""
    pipeline = TransNetV2Pipeline(model_path=args.model_path, device=args.device)
    ocr_pipeline = None
    person_pipeline = None
    transcriber = None
    success = 0

    for index, (video_path, movie_name) in enumerate(videos, start=1):
        movie_output_dir = os.path.join(args.output_dir, movie_name)
        os.makedirs(movie_output_dir, exist_ok=True)

        print(f"\n[{index}/{len(videos)}] 开始处理：{movie_name}")
        print(f"视频路径：{video_path}")
        print(f"输出目录：{movie_output_dir}")

        try:
            pipeline.read_video(video_path)
            shot_count = pipeline.shot_detect(
                output_dir=movie_output_dir,
                threshold=args.threshold,
                min_frames=args.min_frames,
                segment_frames=args.segment_frames,
                save_mp4=not args.no_save_mp4,
            )
            success += 1
            print(f"完成：{movie_name}，共检测到 {shot_count} 个镜头")

            if not args.no_save_mp4 and (not args.no_ocr_check or not args.no_person_check):
                try:
                    if ocr_pipeline is None and not args.no_ocr_check:
                        print("正在初始化 EasyOCR 片头片尾检测模型...")
                        ocr_pipeline = EasyOCRCreditDetectorPipeline(
                            sample_interval_sec=1.0, min_text_boxes=1
                        )
                    if person_pipeline is None and not args.no_person_check:
                        print("正在初始化人物检测模型...")
                        person_pipeline = PersonDetectionPipeline(
                            sample_interval_sec=1.0,
                            min_person_frames=1,
                            confidence_threshold=0.5,
                            device=args.device,
                        )
                    check_credits_and_persons_for_movie(
                        movie_output_dir,
                        movie_name,
                        ocr_pipeline,
                        person_pipeline,
                        args.no_ocr_check,
                        args.no_person_check,
                    )
                except Exception as exc:
                    print(f"OCR/人物检测失败：{movie_name}，错误：{exc}")
                    if not args.continue_on_error:
                        raise

            if not args.no_save_mp4 and not args.no_whisperx:
                try:
                    hf_token = os.getenv("HUGGINGFACE_TOKEN")
                    if hf_token:
                        if transcriber is None:
                            print("正在初始化 WhisperX 转录模型...")
                            wx_device = "cuda" if args.device.startswith("cuda") else args.device
                            transcriber = MovieTranscriber(hf_token=hf_token, device=wx_device)
                        run_whisperx_for_movie(movie_output_dir, transcriber)
                    else:
                        print(f"跳过 WhisperX：未设置 HUGGINGFACE_TOKEN")
                except Exception as exc:
                    print(f"WhisperX 失败：{movie_name}，错误：{exc}")
                    if not args.continue_on_error:
                        raise
        except Exception as exc:
            print(f"失败：{movie_name}，错误：{exc}")
            if not args.continue_on_error:
                raise

    return success


def _run_multi_gpu(args, videos: list, gpu_ids: list[int]) -> int:
    """多卡并行处理：每张 GPU 一个 worker 进程。"""
    ctx = multiprocessing.get_context("spawn")
    work_queue = ctx.Queue()
    result_queue = ctx.Queue()

    worker_args = {
        "output_dir": args.output_dir,
        "model_path": args.model_path,
        "threshold": args.threshold,
        "min_frames": args.min_frames,
        "segment_frames": args.segment_frames,
        "save_mp4": not args.no_save_mp4,
        "no_ocr_check": args.no_ocr_check,
        "no_person_check": args.no_person_check,
        "no_whisperx": args.no_whisperx,
    }

    for video_path, movie_name in videos:
        work_queue.put((video_path, movie_name))
    for _ in gpu_ids:
        work_queue.put(None)

    processes = []
    for gpu_id in gpu_ids:
        env_backup = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        p = ctx.Process(
            target=_worker_process,
            args=(gpu_id, work_queue, result_queue, worker_args),
        )
        p.start()
        processes.append(p)
        if env_backup is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = env_backup
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    success = 0
    with tqdm(total=len(videos), desc="多卡处理") as pbar:
        for _ in range(len(videos)):
            movie_name, ok, val = result_queue.get()
            if ok:
                success += 1
                pbar.set_postfix_str(f"完成: {movie_name}, {val} 镜头")
            else:
                pbar.set_postfix_str(f"失败: {movie_name}")
                print(f"\n失败：{movie_name}，错误：{val}")
                if not args.continue_on_error:
                    for proc in processes:
                        proc.terminate()
                    raise RuntimeError(f"处理失败：{movie_name}")
            pbar.update(1)

    for proc in processes:
        proc.join()

    return success


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.movies_dir):
        print(f"错误：Movies 目录不存在：{args.movies_dir}")
        sys.exit(1)

    videos = collect_mp4_files(args.movies_dir)
    if not videos:
        print(f"未在目录中找到 mp4：{args.movies_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"找到 {len(videos)} 个电影文件")
    print(f"输入目录：{args.movies_dir}")
    print(f"输出目录：{args.output_dir}")

    gpu_ids: list[int] | None = None
    if args.gpus:
        if args.gpus.strip().lower() == "all":
            gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        else:
            gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
        if gpu_ids and torch.cuda.is_available():
            n_dev = torch.cuda.device_count()
            invalid = [i for i in gpu_ids if i < 0 or i >= n_dev]
            if invalid:
                print(f"错误：GPU ID {invalid} 无效，当前仅有 {n_dev} 张卡 (0-{n_dev-1})")
                sys.exit(1)
    if gpu_ids and torch.cuda.is_available():
        print(f"多卡模式：使用 GPU {gpu_ids}")
        print("-" * 60)
        success = _run_multi_gpu(args, videos, gpu_ids)
    else:
        print("单卡模式")
        print("-" * 60)
        print("正在初始化 TransNetV2 模型...")
        success = _run_single_gpu(args, videos)

    print("\n批量处理结束")
    print(f"成功处理：{success}/{len(videos)}")
    print(f"结果位于：{args.output_dir}")


if __name__ == "__main__":
    main()
