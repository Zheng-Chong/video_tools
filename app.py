"""
伺服模式：预处理模型常驻内存，各处理阶段独立轮询各自的输入源并处理。
- TransNet：轮询 movies-dir，处理未做镜头检测的 MP4
- OCR+Person：轮询 output-dir，处理有 shots/ 但缺 clips_check 的
- WhisperX：轮询 output-dir，处理有 shots/ 但缺字幕的
- AVSE：轮询 output-dir，处理有 shots/ 但缺干净语音的
- ASD：轮询 output-dir，处理有 shots+subtitles 但缺 ASD 结果的
每个阶段在每张 GPU 上各有一个 worker 进程，充分利用多卡。
"""
import argparse
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
import warnings

import torch
from tqdm import tqdm

# 抑制 WhisperX 等库的中间日志（ERROR 及以上才输出，屏蔽 WARNING 如 "No active speech found"）
for _name in ("tools.whisperx_pipeline", "whisperx", "whisperx.asr", "whisperx.vads", "whisperx.vads.pyannote", "pyannote", "pyannote.audio", "lightning"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# 抑制 PyTorch/NumPy 在无语音片段上的统计警告
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

from tools.ocr_pipeline import EasyOCRCreditDetectorPipeline
from tools.person_detection_pipeline import PersonDetectionPipeline
from tools.transnet_pipeline import TransNetV2Pipeline
from tools.whisperx_pipeline import MovieTranscriber
from tools.light_asd_pipeline import (
    LightASDPipeline,
    align_subtitles_with_speakers,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MOVIES_DIR = os.path.join(PROJECT_ROOT, "Datasets", "Movies")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Datasets", "AVAGen")

# 全局标志，用于优雅退出（主进程）
_shutdown_requested = False

# 锁文件目录，用于多 worker 避免重复处理
LOCKS_DIR = ".locks"


def _lock_path(output_dir: str, movie_name: str, stage: str) -> str:
    return os.path.join(output_dir, LOCKS_DIR, f"{movie_name}-{stage}.lock")


def _try_acquire_lock(output_dir: str, movie_name: str, stage: str) -> bool:
    """尝试获取锁，成功返回 True，已被占用返回 False"""
    lock_dir = os.path.join(output_dir, LOCKS_DIR)
    os.makedirs(lock_dir, exist_ok=True)
    lock_file = _lock_path(output_dir, movie_name, stage)
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _release_lock(output_dir: str, movie_name: str, stage: str) -> None:
    lock_file = _lock_path(output_dir, movie_name, stage)
    try:
        os.unlink(lock_file)
    except OSError:
        pass


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


def _has_shots_list(output_dir: str, movie_name: str) -> bool:
    """判断是否已完成 TransNet 镜头检测（含 # COMPLETE 标记，避免处理中时被 OCR/WhisperX 抢占）"""
    shots_list = os.path.join(output_dir, movie_name, "shots_list.txt")
    if not os.path.isfile(shots_list):
        return False
    try:
        with open(shots_list, "r") as f:
            lines = [l.strip() for l in f.read().splitlines() if l.strip()]
        return lines and lines[-1] == "# COMPLETE"
    except Exception:
        return False


def _has_shots_dir(output_dir: str, movie_name: str) -> bool:
    """判断是否有 shots 目录"""
    shots_dir = os.path.join(output_dir, movie_name, "shots")
    return os.path.isdir(shots_dir) and any(
        f.lower().endswith(".mp4") for f in os.listdir(shots_dir)
    )


def _ocr_check_complete(movie_output_dir: str, no_ocr: bool, no_person: bool) -> bool:
    """判断 OCR+Person 检测是否已完成"""
    jsonl_path = os.path.join(movie_output_dir, "clips_check.jsonl")
    if not os.path.isfile(jsonl_path):
        return False
    cached = _load_existing_clips_check(jsonl_path)
    shots_dir = os.path.join(movie_output_dir, "shots")
    mp4_files = [f for f in os.listdir(shots_dir) if f.lower().endswith(".mp4")] if os.path.isdir(shots_dir) else []
    return all(
        _record_is_complete(cached.get(f, {}), no_ocr, no_person)
        for f in mp4_files
    )


def _whisperx_complete(movie_output_dir: str) -> bool:
    """判断 WhisperX 是否已完成（所有 shot 都有对应字幕）"""
    shots_dir = os.path.join(movie_output_dir, "shots")
    subtitles_dir = os.path.join(movie_output_dir, "subtitles")
    if not os.path.isdir(shots_dir) or not os.path.isdir(subtitles_dir):
        return False
    mp4_files = [f for f in os.listdir(shots_dir) if f.lower().endswith(".mp4")]
    for f in mp4_files:
        base = os.path.splitext(f)[0]
        if not os.path.isfile(os.path.join(subtitles_dir, f"{base}_whisperx.json")):
            return False
    return True


def _avse_complete(movie_output_dir: str) -> bool:
    """判断 AVSE 是否已完成（仅对 WhisperX 检测到说话的 clip 要求存在 clean wav）"""
    ready, speaking_mp4_files = _get_speaking_clips_from_whisperx(movie_output_dir)
    if not ready:
        return False

    # 没有说话片段时，AVSE 视为完成（无需处理）
    if not speaking_mp4_files:
        return True

    avse_dir = os.path.join(movie_output_dir, "avse")
    if not os.path.isdir(avse_dir):
        return False

    for filename in speaking_mp4_files:
        base = os.path.splitext(filename)[0]
        wav_path = os.path.join(avse_dir, f"{base}_clean.wav")
        if not os.path.isfile(wav_path):
            return False
    return True


def _collect_asd_targets(movie_output_dir: str) -> list[tuple[str, str, str, str]]:
    """
    收集可执行 ASD 的镜头：
    返回 [(shot_path, base_name, srt_path, whisper_json_path), ...]
    仅保留满足以下条件的片段：
    1) clips_check 里 has_person=True 且 has_text=False（有人物且无文字）
    2) 同时存在 whisperx srt + whisperx json
    """
    shots_dir = os.path.join(movie_output_dir, "shots")
    subtitles_dir = os.path.join(movie_output_dir, "subtitles")
    clips_check_jsonl = os.path.join(movie_output_dir, "clips_check.jsonl")
    if not os.path.isdir(shots_dir) or not os.path.isdir(subtitles_dir) or not os.path.isfile(clips_check_jsonl):
        return []

    clip_records = _load_existing_clips_check(clips_check_jsonl)
    targets: list[tuple[str, str, str, str]] = []
    for filename in sorted(f for f in os.listdir(shots_dir) if f.lower().endswith(".mp4")):
        rec = clip_records.get(filename, {})
        if rec.get("has_person") is not True:
            continue
        if rec.get("has_text") is not False:
            continue

        base = os.path.splitext(filename)[0]
        shot_path = os.path.join(shots_dir, filename)
        srt_path = os.path.join(subtitles_dir, f"{base}_whisperx.srt")
        json_path = os.path.join(subtitles_dir, f"{base}_whisperx.json")
        # ASD 仅处理同时具备 srt + json 的片段
        if not os.path.isfile(srt_path) or not os.path.isfile(json_path):
            continue
        targets.append((shot_path, base, srt_path, json_path))
    return targets


def _asd_complete(movie_output_dir: str) -> bool:
    """
    判断 ASD 是否已完成：
    对每个可处理镜头，要求存在：
    1) all_tracks_with_dialog.mp4（带字幕 track 视频）
    2) portraits_yolo/ 至少一张图（YOLO 人物结果）
    3) person_subtitle_mapping.json（人物-字幕对应）
    """
    targets = _collect_asd_targets(movie_output_dir)
    if not targets:
        return False

    asd_root = os.path.join(movie_output_dir, "asd")
    for _, base, _, _ in targets:
        ws_dir = os.path.join(asd_root, base)
        mp4_with_dialog = os.path.join(ws_dir, "pytracks", "all_tracks_with_dialog.mp4")
        portraits_dir = os.path.join(ws_dir, "portraits_yolo")
        mapping_json = os.path.join(ws_dir, "results", "person_subtitle_mapping.json")
        asd_result_json = os.path.join(ws_dir, "asd_results.json")

        # 若预检已明确无可追踪人脸，视为该片段 ASD 已完成（跳过）
        if os.path.isfile(asd_result_json):
            try:
                with open(asd_result_json, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("skip_reason") == "precheck_no_trackable_face":
                    continue
            except Exception:
                pass

        if not os.path.isfile(mp4_with_dialog):
            return False
        if not os.path.isfile(mapping_json):
            return False
        if not os.path.isdir(portraits_dir):
            return False
        if not any(name.lower().endswith(".jpg") for name in os.listdir(portraits_dir)):
            return False
    return True


def _get_speaking_clips_from_whisperx(movie_output_dir: str) -> tuple[bool, list[str]]:
    """
    根据 WhisperX 结果返回需要做 AVSE 的片段：
    - 返回 (ready, speaking_mp4_files)
    - ready=False 表示 WhisperX 结果尚不完整（例如仍缺字幕 json）
    - speaking_mp4_files 为检测到有语音内容的 mp4 文件名列表
    """
    shots_dir = os.path.join(movie_output_dir, "shots")
    subtitles_dir = os.path.join(movie_output_dir, "subtitles")
    if not os.path.isdir(shots_dir) or not os.path.isdir(subtitles_dir):
        return False, []

    mp4_files = sorted(f for f in os.listdir(shots_dir) if f.lower().endswith(".mp4"))
    speaking_files: list[str] = []

    for filename in mp4_files:
        base = os.path.splitext(filename)[0]
        whisper_json = os.path.join(subtitles_dir, f"{base}_whisperx.json")
        if not os.path.isfile(whisper_json):
            return False, []

        try:
            with open(whisper_json, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            # 字幕文件损坏/不可读时，当作“无语音”跳过 AVSE，避免卡住轮询
            continue

        segments = result.get("segments")
        if isinstance(segments, list) and len(segments) > 0:
            speaking_files.append(filename)

    return True, speaking_files


# --- 阶段 1：TransNet 镜头检测 ---
def get_pending_transnet(movies_dir: str, output_dir: str) -> list[tuple[str, str]]:
    """待 TransNet 处理：movies_dir 下的 MP4，且 output_dir 中尚无 shots_list.txt"""
    all_videos = collect_mp4_files(movies_dir)
    return [
        (video_path, movie_name)
        for video_path, movie_name in all_videos
        if not _has_shots_list(output_dir, movie_name)
    ]


# --- 阶段 2：OCR + Person 检测 ---
def get_pending_ocr_person(
    output_dir: str, no_ocr_check: bool, no_person_check: bool
) -> list[tuple[str, str]]:
    """待 OCR+Person 处理：output_dir 下 TransNet 已完成且有 shots/ 的目录，且 clips_check 缺失或不完整"""
    if no_ocr_check and no_person_check:
        return []
    pending: list[tuple[str, str]] = []
    for name in os.listdir(output_dir):
        movie_dir = os.path.join(output_dir, name)
        if not os.path.isdir(movie_dir):
            continue
        if not _has_shots_list(output_dir, name) or not _has_shots_dir(output_dir, name):
            continue
        if _ocr_check_complete(movie_dir, no_ocr_check, no_person_check):
            continue
        pending.append((movie_dir, name))
    return sorted(pending, key=lambda x: x[1])


# --- 阶段 3：WhisperX 字幕 ---
def get_pending_whisperx(output_dir: str) -> list[tuple[str, str]]:
    """待 WhisperX 处理：output_dir 下 TransNet 已完成且有 shots/ 的目录，且 subtitles 缺失或不完整"""
    pending: list[tuple[str, str]] = []
    for name in os.listdir(output_dir):
        movie_dir = os.path.join(output_dir, name)
        if not os.path.isdir(movie_dir):
            continue
        if not _has_shots_list(output_dir, name) or not _has_shots_dir(output_dir, name):
            continue
        if _whisperx_complete(movie_dir):
            continue
        pending.append((movie_dir, name))
    return sorted(pending, key=lambda x: x[1])


# --- 阶段 4：AVSE 干净语音 ---
def get_pending_avse(output_dir: str) -> list[tuple[str, str]]:
    """待 AVSE 处理：output_dir 下 TransNet 已完成且有 shots/ 的目录，且 avse 缺失或不完整"""
    pending: list[tuple[str, str]] = []
    for name in os.listdir(output_dir):
        movie_dir = os.path.join(output_dir, name)
        if not os.path.isdir(movie_dir):
            continue
        if not _has_shots_list(output_dir, name) or not _has_shots_dir(output_dir, name):
            continue
        # AVSE 依赖 WhisperX 的说话检测结果；WhisperX 未完成时不进入 AVSE 阶段
        if not _whisperx_complete(movie_dir):
            continue
        if _avse_complete(movie_dir):
            continue
        pending.append((movie_dir, name))
    return sorted(pending, key=lambda x: x[1])


# --- 阶段 5：ASD 说话人跟踪 ---
def get_pending_asd(output_dir: str) -> list[tuple[str, str]]:
    """
    待 ASD 处理：output_dir 下 TransNet+WhisperX 已完成且有 shots 的目录，且 ASD 结果不完整。
    """
    pending: list[tuple[str, str]] = []
    for name in os.listdir(output_dir):
        movie_dir = os.path.join(output_dir, name)
        if not os.path.isdir(movie_dir):
            continue
        if not _has_shots_list(output_dir, name) or not _has_shots_dir(output_dir, name):
            continue
        if not _whisperx_complete(movie_dir):
            continue
        # 没有满足“有人物且无文字且有 srt+json”的片段，则无需进入 ASD 阶段
        if not _collect_asd_targets(movie_dir):
            continue
        if _asd_complete(movie_dir):
            continue
        pending.append((movie_dir, name))
    return sorted(pending, key=lambda x: x[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="伺服模式：模型常驻，轮询待处理 MP4 并自动处理"
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
        help="TransNetV2 模型目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备（多卡时由 --gpus 指定，每 worker 绑定单卡）",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="all",
        help='使用的 GPU ID，逗号分隔如 "0,1,2,3"，或 "all" 使用全部可用 GPU',
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="轮询间隔（秒），无待处理任务时每隔多久检查一次",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="镜头检测阈值",
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
        help="不保存切分后的镜头 MP4",
    )
    parser.add_argument(
        "--no-ocr-check",
        action="store_true",
        help="不进行片头片尾 OCR 检测",
    )
    parser.add_argument(
        "--no-person-check",
        action="store_true",
        help="不进行人物检测",
    )
    parser.add_argument(
        "--no-whisperx",
        action="store_true",
        help="不进行 WhisperX 字幕转录",
    )
    parser.add_argument(
        "--no-avse",
        action="store_true",
        help="不进行 AVSE 干净语音提取",
    )
    parser.add_argument(
        "--no-asd",
        action="store_true",
        help="不进行 ASD 说话人跟踪/字幕对齐/人物图导出",
    )
    parser.add_argument(
        "--asd-threshold",
        type=float,
        default=-0.4,
        help="ASD 说话判定阈值",
    )
    parser.add_argument(
        "--asd-face-conf-th",
        type=float,
        default=0.75,
        help="ASD 人脸检测阈值（降低可提升召回）",
    )
    parser.add_argument(
        "--asd-min-track-frames",
        type=int,
        default=4,
        help="ASD 最小轨迹帧数",
    )
    parser.add_argument(
        "--asd-min-shot-sec",
        type=float,
        default=0.4,
        help="ASD 最小镜头时长（秒）",
    )
    parser.add_argument(
        "--asd-precheck-face-conf-th",
        type=float,
        default=0.55,
        help="ASD 预检人脸阈值",
    )
    parser.add_argument(
        "--asd-precheck-sample-interval-sec",
        type=float,
        default=0.4,
        help="ASD 预检采样间隔（秒）",
    )
    parser.add_argument(
        "--asd-precheck-min-face-hits",
        type=int,
        default=1,
        help="ASD 预检命中最少人脸帧数",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="某个电影失败时继续处理下一个",
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
    """判断已有记录是否完整"""
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
    """对 shots/ 中的片段进行 OCR 和人物检测，保存到 clips_check.jsonl"""
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
    print(f"检测完成：{credits_count} 个有文字，{persons_count} 个有人物，结果已保存：{jsonl_path}")


def run_whisperx_for_movie(
    movie_output_dir: str,
    transcriber: MovieTranscriber,
) -> None:
    """对 shots/ 中的片段运行 WhisperX 转录，保存到 subtitles/。使用 process_many 批量处理，模型只加载一次。已存在 *_whisperx.json 的片段跳过。"""
    shots_dir = os.path.join(movie_output_dir, "shots")
    subtitles_dir = os.path.join(movie_output_dir, "subtitles")
    if not os.path.isdir(shots_dir):
        return

    all_mp4 = sorted(
        f for f in os.listdir(shots_dir) if f.lower().endswith(".mp4")
    )
    if not all_mp4:
        return

    os.makedirs(subtitles_dir, exist_ok=True)

    # 片段级别跳过：仅处理尚无 *_whisperx.json 的片段
    mp4_files = [
        f for f in all_mp4
        if not os.path.isfile(os.path.join(subtitles_dir, f"{os.path.splitext(f)[0]}_whisperx.json"))
    ]
    if not mp4_files:
        print(f"WhisperX 已全部完成：{subtitles_dir}")
        return

    skipped = len(all_mp4) - len(mp4_files)
    if skipped:
        print(f"WhisperX 跳过 {skipped} 个已处理片段，待处理 {len(mp4_files)} 个")

    video_paths = [os.path.join(shots_dir, f) for f in mp4_files]

    try:
        results = transcriber.process_many(
            input_paths=video_paths,
            batch_size=16,
            language="en",
        )
        for filename, result in zip(mp4_files, results):
            base_name = os.path.splitext(filename)[0]
            out_prefix = os.path.join(subtitles_dir, f"{base_name}_whisperx")
            json_path = f"{out_prefix}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            transcriber.save_srt(result, f"{out_prefix}.srt")
        print(f"WhisperX 字幕已保存至：{subtitles_dir}（本批 {len(results)} 个片段）")
    except Exception as exc:
        print(f"WhisperX 批量失败：{exc}")
        # 降级为逐片段处理
        for filename in tqdm(mp4_files, desc="WhisperX 转录（降级）"):
            base_name = os.path.splitext(filename)[0]
            out_prefix = os.path.join(subtitles_dir, f"{base_name}_whisperx")
            if os.path.isfile(f"{out_prefix}.json"):
                continue
            video_path = os.path.join(shots_dir, filename)
            try:
                result = transcriber.process(input_path=video_path, batch_size=16, language="en")
                with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                transcriber.save_srt(result, f"{out_prefix}.srt")
            except Exception as e:
                print(f"WhisperX 失败 {filename}: {e}")
                error_result = {"error": str(e), "segments": []}
                with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
                    json.dump(error_result, f, ensure_ascii=False, indent=2)


def run_avse_for_movie(
    movie_output_dir: str,
    avse_model,
    process_clip_fn,
) -> None:
    """对 WhisperX 判定有语音的片段运行 AVSE，输出到 avse/。已存在 *_clean.wav 的片段会跳过。"""
    avse_dir = os.path.join(movie_output_dir, "avse")

    ready, speaking_mp4_files = _get_speaking_clips_from_whisperx(movie_output_dir)
    if not ready:
        print(f"AVSE 等待 WhisperX 完成：{movie_output_dir}")
        return

    if not speaking_mp4_files:
        print(f"AVSE 无需处理（WhisperX 未检测到语音片段）：{movie_output_dir}")
        return

    os.makedirs(avse_dir, exist_ok=True)
    pending_files = []
    for filename in speaking_mp4_files:
        base_name = os.path.splitext(filename)[0]
        output_wav = os.path.join(avse_dir, f"{base_name}_clean.wav")
        # 片段级别去重：已处理则跳过，未处理才送入 AVSE 模型
        if os.path.isfile(output_wav):
            continue
        pending_files.append(filename)

    if not pending_files:
        print(f"AVSE 已全部完成：{avse_dir}")
        return

    skipped = len(speaking_mp4_files) - len(pending_files)
    if skipped:
        print(f"AVSE 跳过 {skipped} 个已处理片段，待处理 {len(pending_files)} 个")

    shots_dir = os.path.join(movie_output_dir, "shots")
    for filename in tqdm(pending_files, desc="AVSE 提取"):
        video_path = os.path.join(shots_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_wav = os.path.join(avse_dir, f"{base_name}_clean.wav")
        try:
            process_clip_fn(
                input_mp4_path=video_path,
                output_clean_wav_path=output_wav,
                avse_model=avse_model,
            )
        except Exception as exc:
            print(f"AVSE 失败 {filename}: {exc}")


def run_asd_for_movie(
    movie_output_dir: str,
    asd_pipeline: LightASDPipeline,
    asd_threshold: float,
) -> None:
    """
    对每个 shot 执行 ASD，并输出：
    - asd/<shot>/pytracks/all_tracks_with_dialog.mp4
    - asd/<shot>/portraits_yolo/*.jpg
    - asd/<shot>/results/person_subtitle_mapping.json
    同时保留 subtitles/<shot>_whisperx_speaker_aligned.json 便于全局检索。
    """
    targets = _collect_asd_targets(movie_output_dir)
    if not targets:
        print(f"ASD 无可处理片段：{movie_output_dir}")
        return

    asd_root = os.path.join(movie_output_dir, "asd")
    subtitles_dir = os.path.join(movie_output_dir, "subtitles")
    os.makedirs(asd_root, exist_ok=True)

    for shot_path, base, srt_path, json_path in tqdm(targets, desc="ASD 对齐"):
        ws_dir = os.path.join(asd_root, base)
        result_dir = os.path.join(ws_dir, "results")
        pytracks_dir = os.path.join(ws_dir, "pytracks")
        portraits_dir = os.path.join(ws_dir, "portraits_yolo")
        os.makedirs(result_dir, exist_ok=True)

        out_mp4 = os.path.join(pytracks_dir, "all_tracks_with_dialog.mp4")
        mapping_json = os.path.join(result_dir, "person_subtitle_mapping.json")
        aligned_json_in_subtitles = os.path.join(subtitles_dir, f"{base}_whisperx_speaker_aligned.json")

        done = (
            os.path.isfile(out_mp4)
            and os.path.isfile(mapping_json)
            and os.path.isdir(portraits_dir)
            and any(name.lower().endswith(".jpg") for name in os.listdir(portraits_dir))
        )
        if done:
            continue

        try:
            vid_tracks, scores = asd_pipeline.process_video(
                video_path=shot_path,
                workspace_base=asd_root,
                threshold=asd_threshold,
            )
        except Exception as exc:
            print(f"ASD 主流程失败 {base}: {exc}")
            continue

        subtitle_input = srt_path or json_path
        aligned = []
        if subtitle_input:
            try:
                aligned = align_subtitles_with_speakers(
                    vid_tracks,
                    scores,
                    subtitle_path=subtitle_input,
                    word_json_path=json_path,
                    fps=25,
                    threshold=asd_threshold,
                )
            except Exception as exc:
                print(f"ASD 字幕对齐失败 {base}: {exc}")
                aligned = []

        # 保存人物-字幕对应 JSON（按你要求固定到 ASD 目录）
        try:
            with open(mapping_json, "w", encoding="utf-8") as f:
                json.dump(aligned, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"保存人物-字幕对应 JSON 失败 {base}: {exc}")

        # 再冗余保存一份到 subtitles 目录（便于与 whisperx 同目录查看）
        if aligned:
            try:
                with open(aligned_json_in_subtitles, "w", encoding="utf-8") as f:
                    json.dump(aligned, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                print(f"保存 subtitles 对齐 JSON 失败 {base}: {exc}")

        # 导出 YOLO 人物图
        try:
            asd_pipeline.export_speaker_portraits_yolo(shot_path, vid_tracks, portraits_dir)
        except Exception as exc:
            print(f"导出 YOLO 人物图失败 {base}: {exc}")

        # 导出带字幕 track 视频
        if aligned and asd_pipeline.last_track_mp4_paths:
            try:
                all_tracks_mp4 = asd_pipeline.last_track_mp4_paths[0]
                audio_file_path = os.path.join(ws_dir, "pyavi", "audio.wav")
                asd_pipeline.render_aligned_subtitles_on_video(
                    input_video_path=all_tracks_mp4,
                    audio_file_path=audio_file_path,
                    aligned_subtitles=aligned,
                    output_mp4_path=out_mp4,
                    fps=25,
                )
            except Exception as exc:
                print(f"导出带字幕 track MP4 失败 {base}: {exc}")


def run_stage_transnet(
    video_path: str,
    movie_name: str,
    output_dir: str,
    pipeline: TransNetV2Pipeline,
    args: argparse.Namespace,
) -> bool:
    """阶段 1：仅执行 TransNet 镜头检测"""
    movie_output_dir = os.path.join(output_dir, movie_name)
    os.makedirs(movie_output_dir, exist_ok=True)
    try:
        pipeline.read_video(video_path)
        shot_count = pipeline.shot_detect(
            output_dir=movie_output_dir,
            threshold=args.threshold,
            min_frames=args.min_frames,
            segment_frames=args.segment_frames,
            save_mp4=not args.no_save_mp4,
        )
        print(f"[TransNet] 完成：{movie_name}，共 {shot_count} 个镜头")
        return True
    except Exception as exc:
        print(f"[TransNet] 失败：{movie_name}，错误：{exc}")
        return False


def run_stage_ocr_person(
    movie_output_dir: str,
    movie_name: str,
    ocr_pipeline: EasyOCRCreditDetectorPipeline | None,
    person_pipeline: PersonDetectionPipeline | None,
    no_ocr_check: bool,
    no_person_check: bool,
) -> bool:
    """阶段 2：仅执行 OCR + 人物检测"""
    try:
        check_credits_and_persons_for_movie(
            movie_output_dir,
            movie_name,
            ocr_pipeline,
            person_pipeline,
            no_ocr_check,
            no_person_check,
        )
        return True
    except Exception as exc:
        print(f"[OCR+Person] 失败：{movie_name}，错误：{exc}")
        return False


def run_stage_whisperx(
    movie_output_dir: str,
    movie_name: str,
    transcriber: MovieTranscriber,
) -> bool:
    """阶段 3：仅执行 WhisperX 字幕转录"""
    try:
        run_whisperx_for_movie(movie_output_dir, transcriber)
        return True
    except Exception as exc:
        print(f"[WhisperX] 失败：{movie_name}，错误：{exc}")
        return False


def run_stage_avse(
    movie_output_dir: str,
    movie_name: str,
    avse_model,
    process_clip_fn,
) -> bool:
    """阶段 4：仅执行 AVSE 干净语音提取"""
    try:
        run_avse_for_movie(
            movie_output_dir=movie_output_dir,
            avse_model=avse_model,
            process_clip_fn=process_clip_fn,
        )
        return True
    except Exception as exc:
        print(f"[AVSE] 失败：{movie_name}，错误：{exc}")
        return False


def run_stage_asd(
    movie_output_dir: str,
    movie_name: str,
    asd_pipeline: LightASDPipeline,
    asd_threshold: float,
) -> bool:
    """阶段 5：仅执行 ASD 说话人跟踪及结果导出"""
    try:
        run_asd_for_movie(
            movie_output_dir=movie_output_dir,
            asd_pipeline=asd_pipeline,
            asd_threshold=asd_threshold,
        )
        return True
    except Exception as exc:
        print(f"[ASD] 失败：{movie_name}，错误：{exc}")
        return False


def _worker_transnet(
    gpu_id: int | None,
    args: argparse.Namespace,
    shutdown_event: multiprocessing.Event,
) -> None:
    """TransNet worker：绑定 GPU（或 CPU），仅加载 TransNet 模型，轮询并处理"""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda"
    else:
        device = "cpu"

    dev_label = f"GPU{gpu_id}" if gpu_id is not None else "CPU"
    print(f"[TransNet {dev_label}] 正在加载模型...")
    transnet = TransNetV2Pipeline(model_path=args.model_path, device=device)
    print(f"[TransNet {dev_label}] 模型加载完成，开始轮询")
    stage = "transnet"

    while not shutdown_event.is_set():
        pending = get_pending_transnet(args.movies_dir, args.output_dir)
        if pending:
            for video_path, movie_name in pending:
                if shutdown_event.is_set():
                    return
                if not _try_acquire_lock(args.output_dir, movie_name, stage):
                    continue
                try:
                    print(f"[TransNet {dev_label}] 处理：{movie_name}")
                    ok = run_stage_transnet(
                        video_path, movie_name, args.output_dir, transnet, args
                    )
                    if not ok and not args.continue_on_error:
                        return
                finally:
                    _release_lock(args.output_dir, movie_name, stage)
                break
        else:
            for _ in range(args.poll_interval):
                if shutdown_event.is_set():
                    return
                time.sleep(1)


def _worker_ocr_person(
    gpu_id: int | None,
    args: argparse.Namespace,
    shutdown_event: multiprocessing.Event,
) -> None:
    """OCR+Person worker：绑定 GPU（或 CPU），仅加载 OCR 和 Person 模型，轮询并处理"""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda"
    else:
        device = "cpu"

    dev_label = f"GPU{gpu_id}" if gpu_id is not None else "CPU"
    print(f"[OCR+Person {dev_label}] 正在加载模型...")

    ocr = None
    if not args.no_ocr_check:
        ocr = EasyOCRCreditDetectorPipeline(
            sample_interval_sec=1.0,
            min_text_boxes=1,
        )

    person = None
    if not args.no_person_check:
        person = PersonDetectionPipeline(
            sample_interval_sec=1.0,
            min_person_frames=1,
            confidence_threshold=0.5,
            device=device,
        )

    if not ocr and not person:
        return

    print(f"[OCR+Person {dev_label}] 模型加载完成，开始轮询")
    stage = "ocr_person"

    while not shutdown_event.is_set():
        if args.no_save_mp4:
            time.sleep(args.poll_interval)
            continue
        pending = get_pending_ocr_person(
            args.output_dir, args.no_ocr_check, args.no_person_check
        )
        if pending:
            for movie_dir, movie_name in pending:
                if shutdown_event.is_set():
                    return
                if not _try_acquire_lock(args.output_dir, movie_name, stage):
                    continue
                try:
                    print(f"[OCR+Person {dev_label}] 处理：{movie_name}")
                    ok = run_stage_ocr_person(
                        movie_dir, movie_name,
                        ocr, person,
                        args.no_ocr_check, args.no_person_check,
                    )
                    if not ok and not args.continue_on_error:
                        return
                finally:
                    _release_lock(args.output_dir, movie_name, stage)
                break
        else:
            for _ in range(args.poll_interval):
                if shutdown_event.is_set():
                    return
                time.sleep(1)


def _worker_whisperx(
    gpu_id: int | None,
    args: argparse.Namespace,
    shutdown_event: multiprocessing.Event,
) -> None:
    """WhisperX worker：绑定 GPU（或 CPU），仅加载 WhisperX 模型，轮询并处理"""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda"
    else:
        device = "cpu"

    dev_label = f"GPU{gpu_id}" if gpu_id is not None else "CPU"
    wx_device = "cuda" if device.startswith("cuda") else device
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    if not hf_token:
        print(f"[WhisperX {dev_label}] 未设置 HUGGINGFACE_TOKEN，退出")
        return

    print(f"[WhisperX {dev_label}] 正在加载模型（Whisper + Align + Diarize）...")
    transcriber = MovieTranscriber(
        hf_token=hf_token,
        device=wx_device,
    )
    print(f"[WhisperX {dev_label}] 所有模型加载完成，开始轮询")
    stage = "whisperx"

    while not shutdown_event.is_set():
        if args.no_save_mp4:
            time.sleep(args.poll_interval)
            continue
        pending = get_pending_whisperx(args.output_dir)
        if pending:
            for movie_dir, movie_name in pending:
                if shutdown_event.is_set():
                    return
                if not _try_acquire_lock(args.output_dir, movie_name, stage):
                    continue
                try:
                    print(f"[WhisperX {dev_label}] 处理：{movie_name}")
                    ok = run_stage_whisperx(movie_dir, movie_name, transcriber)
                    if not ok and not args.continue_on_error:
                        return
                finally:
                    _release_lock(args.output_dir, movie_name, stage)
                break
        else:
            for _ in range(args.poll_interval):
                if shutdown_event.is_set():
                    return
                time.sleep(1)


def _worker_avse(
    gpu_id: int | None,
    args: argparse.Namespace,
    shutdown_event: multiprocessing.Event,
) -> None:
    """AVSE worker：绑定 GPU（或 CPU），仅加载 AVSE 模型，轮询并处理"""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda"
    else:
        device = "cpu"

    dev_label = f"GPU{gpu_id}" if gpu_id is not None else "CPU"
    try:
        from tools.avse_pipeline import build_avse_model, run_avse_clip
    except Exception as exc:
        print(f"[AVSE {dev_label}] 导入失败：{exc}")
        return

    print(f"[AVSE {dev_label}] 正在加载模型...")
    avse_model = build_avse_model()
    print(f"[AVSE {dev_label}] 模型加载完成，开始轮询")
    stage = "avse"

    while not shutdown_event.is_set():
        if args.no_save_mp4:
            time.sleep(args.poll_interval)
            continue
        pending = get_pending_avse(args.output_dir)
        if pending:
            for movie_dir, movie_name in pending:
                if shutdown_event.is_set():
                    return
                if not _try_acquire_lock(args.output_dir, movie_name, stage):
                    continue
                try:
                    print(f"[AVSE {dev_label}] 处理：{movie_name}")
                    ok = run_stage_avse(
                        movie_output_dir=movie_dir,
                        movie_name=movie_name,
                        avse_model=avse_model,
                        process_clip_fn=run_avse_clip,
                    )
                    if not ok and not args.continue_on_error:
                        return
                finally:
                    _release_lock(args.output_dir, movie_name, stage)
                break
        else:
            for _ in range(args.poll_interval):
                if shutdown_event.is_set():
                    return
                time.sleep(1)


def _worker_asd(
    gpu_id: int | None,
    args: argparse.Namespace,
    shutdown_event: multiprocessing.Event,
) -> None:
    """ASD worker：绑定 GPU（或 CPU），加载 Light-ASD，轮询并处理"""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda"
    else:
        device = "cpu"

    dev_label = f"GPU{gpu_id}" if gpu_id is not None else "CPU"
    print(f"[ASD {dev_label}] 正在加载模型...")
    asd_pipeline = LightASDPipeline(
        device=device,
        face_conf_th=args.asd_face_conf_th,
        min_track_frames=args.asd_min_track_frames,
        min_shot_sec=args.asd_min_shot_sec,
        precheck_face_conf_th=args.asd_precheck_face_conf_th,
        precheck_sample_interval_sec=args.asd_precheck_sample_interval_sec,
        precheck_min_face_hits=args.asd_precheck_min_face_hits,
    )
    print(f"[ASD {dev_label}] 模型加载完成，开始轮询")
    stage = "asd"

    while not shutdown_event.is_set():
        if args.no_save_mp4:
            time.sleep(args.poll_interval)
            continue
        pending = get_pending_asd(args.output_dir)
        if pending:
            for movie_dir, movie_name in pending:
                if shutdown_event.is_set():
                    return
                if not _try_acquire_lock(args.output_dir, movie_name, stage):
                    continue
                try:
                    print(f"[ASD {dev_label}] 处理：{movie_name}")
                    ok = run_stage_asd(
                        movie_output_dir=movie_dir,
                        movie_name=movie_name,
                        asd_pipeline=asd_pipeline,
                        asd_threshold=args.asd_threshold,
                    )
                    if not ok and not args.continue_on_error:
                        return
                finally:
                    _release_lock(args.output_dir, movie_name, stage)
                break
        else:
            for _ in range(args.poll_interval):
                if shutdown_event.is_set():
                    return
                time.sleep(1)


def _resolve_gpu_ids(gpus_arg: str) -> list[int | None]:
    """解析 --gpus 参数为 GPU ID 列表，None 表示 CPU"""
    if not gpus_arg or gpus_arg.strip().lower() == "all":
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return [None]  # 无 GPU 时单 worker 用 CPU
    return [int(x.strip()) for x in gpus_arg.split(",") if x.strip()]


def main() -> None:
    global _shutdown_requested

    args = parse_args()

    gpu_ids = _resolve_gpu_ids(args.gpus)
    if not gpu_ids:
        gpu_ids = [None] if not torch.cuda.is_available() else list(range(torch.cuda.device_count()))
    n_gpus = len(gpu_ids)

    if not os.path.isdir(args.movies_dir):
        print(f"错误：Movies 目录不存在：{args.movies_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    ctx = multiprocessing.get_context("spawn")
    shutdown_event = ctx.Event()

    def _set_shutdown(sig=None, frame=None):
        global _shutdown_requested
        _shutdown_requested = True
        shutdown_event.set()
        if sig is not None:
            print("\n收到退出信号，将在当前任务完成后停止...")

    signal.signal(signal.SIGINT, _set_shutdown)
    signal.signal(signal.SIGTERM, _set_shutdown)

    print("=" * 60)
    print("伺服模式启动（每阶段每 GPU 一个 worker）")
    print(f"输入目录：{args.movies_dir}")
    print(f"输出目录：{args.output_dir}")
    gpu_display = [f"GPU{i}" if i is not None else "CPU" for i in gpu_ids]
    print(f"设备：{gpu_display}（每阶段 {n_gpus} 个 worker）")
    print(f"轮询间隔：{args.poll_interval}s")
    print("=" * 60)

    processes: list[tuple[multiprocessing.Process, int | None]] = []

    def _dev_name(gid: int | None) -> str:
        return f"GPU{gid}" if gid is not None else "CPU"

    for gpu_id in gpu_ids:
        p = ctx.Process(
            target=_worker_transnet,
            args=(gpu_id, args, shutdown_event),
            name=f"TransNet-{_dev_name(gpu_id)}",
        )
        processes.append((p, gpu_id))

    if not args.no_ocr_check or not args.no_person_check:
        for gpu_id in gpu_ids:
            p = ctx.Process(
                target=_worker_ocr_person,
                args=(gpu_id, args, shutdown_event),
                name=f"OCR+Person-{_dev_name(gpu_id)}",
            )
            processes.append((p, gpu_id))

    if not args.no_whisperx and os.getenv("HUGGINGFACE_TOKEN"):
        for gpu_id in gpu_ids:
            p = ctx.Process(
                target=_worker_whisperx,
                args=(gpu_id, args, shutdown_event),
                name=f"WhisperX-{_dev_name(gpu_id)}",
            )
            processes.append((p, gpu_id))

    if not args.no_avse:
        for gpu_id in gpu_ids:
            p = ctx.Process(
                target=_worker_avse,
                args=(gpu_id, args, shutdown_event),
                name=f"AVSE-{_dev_name(gpu_id)}",
            )
            processes.append((p, gpu_id))

    if not args.no_asd:
        for gpu_id in gpu_ids:
            p = ctx.Process(
                target=_worker_asd,
                args=(gpu_id, args, shutdown_event),
                name=f"ASD-{_dev_name(gpu_id)}",
            )
            processes.append((p, gpu_id))

    # 启动前设置 CUDA_VISIBLE_DEVICES，子进程继承该环境
    env_backup = os.environ.get("CUDA_VISIBLE_DEVICES")
    for proc, gpu_id in processes:
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        proc.start()
    if env_backup is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = env_backup
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    print(f"已启动 {len(processes)} 个 worker 进程")

    try:
        while not _shutdown_requested:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    shutdown_event.set()
    print("等待 worker 结束...")
    for proc, _ in processes:
        proc.join(timeout=60)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)

    print("\n伺服已停止")


if __name__ == "__main__":
    main()
