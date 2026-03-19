"""视频 IO 模块：FFmpeg 封装等（避免与内置 io 模块冲突）。"""

from .video_utils import (
    extract_video_to_workspace,
    extract_audio_segment,
    iter_sampled_frames,
    mux_video_audio,
    transcode_with_fallback,
)

__all__ = [
    "extract_video_to_workspace",
    "extract_audio_segment",
    "iter_sampled_frames",
    "mux_video_audio",
    "transcode_with_fallback",
]
