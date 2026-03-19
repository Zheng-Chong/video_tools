"""
视频处理基建：FFmpeg 封装，统一 subprocess 调用。
将抽帧、音频提取、转码等逻辑从 pipeline 中抽离。
"""

import os
import subprocess
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


def iter_sampled_frames(
    video_path: str,
    sample_interval_sec: float = 1.0,
    default_fps: float = 24.0,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    按时间间隔采样视频帧的迭代器。
    供 ocr_pipeline、person_detection_pipeline 等复用。

    Args:
        video_path: 视频文件路径
        sample_interval_sec: 采样间隔（秒）
        default_fps: 无法从视频获取 fps 时的默认值

    Yields:
        (sampled_index, frame): 采样序号（从 1 开始）、BGR 格式的 numpy 数组

    Raises:
        ValueError: 无法打开视频文件时
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = default_fps
    frame_interval = max(1, int(fps * sample_interval_sec))
    frame_count = 0
    sampled_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                sampled_index += 1
                yield sampled_index, frame
            frame_count += 1
    finally:
        cap.release()


def extract_video_to_workspace(
    input_video_path: str,
    video_avi_path: str,
    audio_wav_path: str,
    frames_pattern: str,
    n_threads: int = 2,
) -> None:
    """
    从输入视频提取：1) 25fps AVI 2) 16kHz 单声道 WAV 3) 帧序列 JPG。
    """
    subprocess.call(
        f'ffmpeg -y -i "{input_video_path}" -qscale:v 2 -threads {n_threads} '
        f'-async 1 -r 25 "{video_avi_path}" -loglevel panic',
        shell=True,
    )
    subprocess.call(
        f'ffmpeg -y -i "{video_avi_path}" -qscale:a 0 -ac 1 -vn '
        f'-threads {n_threads} -ar 16000 "{audio_wav_path}" -loglevel panic',
        shell=True,
    )
    subprocess.call(
        f'ffmpeg -y -i "{video_avi_path}" -qscale:v 2 -threads {n_threads} '
        f'-f image2 "{frames_pattern}" -loglevel panic',
        shell=True,
    )


def extract_audio_segment(
    audio_path: str,
    output_wav_path: str,
    start_sec: float,
    end_sec: float,
    n_threads: int = 2,
) -> None:
    """从音频文件中截取指定时间段，输出 16kHz 单声道 WAV。"""
    subprocess.call(
        f'ffmpeg -y -i "{audio_path}" -async 1 -ac 1 -vn '
        f'-acodec pcm_s16le -ar 16000 -threads {n_threads} '
        f'-ss {start_sec:.3f} -to {end_sec:.3f} "{output_wav_path}" -loglevel panic',
        shell=True,
    )


def mux_video_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    n_threads: int = 2,
) -> None:
    """将视频与音频封装到同一文件（copy 模式）。"""
    subprocess.call(
        f'ffmpeg -y -i "{video_path}" -i "{audio_path}" -threads {n_threads} '
        f'-c:v copy -c:a copy "{output_path}" -loglevel panic',
        shell=True,
    )


def transcode_with_fallback(
    raw_mp4_path: str,
    audio_file_path: str,
    output_mp4_path: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Tuple[bool, str, str]:
    """
    使用 ffmpeg 转码并封装音频，自动回退视频编码器。
    返回 (成功, 使用的编码器, 错误信息)。
    """
    candidate_video_encoders = ["libx264", "libopenh264", "h264_v4l2m2m", "mpeg4"]
    last_err = ""

    for vcodec in candidate_video_encoders:
        transcode_cmd = ["ffmpeg", "-y"]
        if start_sec is not None and end_sec is not None:
            transcode_cmd.extend(["-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}"])
        transcode_cmd.extend(["-i", raw_mp4_path])
        if start_sec is not None and end_sec is not None:
            transcode_cmd.extend(["-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}"])
        transcode_cmd.extend([
            "-i", audio_file_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-af", "aresample=async=1:first_pts=0",
            "-c:v", vcodec,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-ac", "2",
            "-ar", "48000",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-shortest",
            output_mp4_path,
        ])
        if vcodec == "libx264":
            insert_idx = transcode_cmd.index("-pix_fmt")
            transcode_cmd[insert_idx:insert_idx] = ["-preset", "veryfast"]

        result = subprocess.run(transcode_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True, vcodec, ""

        last_err = (result.stderr or "").strip()
        if os.path.exists(output_mp4_path):
            os.remove(output_mp4_path)

    return False, "", last_err
