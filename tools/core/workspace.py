"""
ASD 工作区管理：统一路径拼接、目录初始化与清理逻辑。
将散落在各处的 pyavi、pyframes、pycrop、pytracks、results、portraits 等路径逻辑集中管理。
"""

import os
import shutil


class ASDWorkspace:
    """
    单个 ASD 视频的工作区，管理所有中间与输出路径。
    """

    def __init__(self, workspace_base: str, video_name: str):
        """
        Args:
            workspace_base: 工作区根目录（如 movie_dir/asd）
            video_name: 视频/镜头名称（不含扩展名）
        """
        self.root = os.path.join(workspace_base, video_name)
        self.pyavi = os.path.join(self.root, "pyavi")
        self.pyframes = os.path.join(self.root, "pyframes")
        self.pycrop = os.path.join(self.root, "pycrop")
        self.pytracks = os.path.join(self.root, "pytracks")
        self.results = os.path.join(self.root, "results")
        self.portraits = os.path.join(self.root, "portraits_yolo")

    @property
    def video_avi(self) -> str:
        """pyavi 下的 video.avi 路径"""
        return os.path.join(self.pyavi, "video.avi")

    @property
    def audio_wav(self) -> str:
        """pyavi 下的 audio.wav 路径"""
        return os.path.join(self.pyavi, "audio.wav")

    @property
    def frames_pattern(self) -> str:
        """抽帧输出路径模式，用于 ffmpeg -f image2"""
        return os.path.join(self.pyframes, "%06d.jpg")

    @property
    def tracking_json(self) -> str:
        """tracking_results.json 路径（在 ws 根目录，与 asd_results 一致）"""
        return os.path.join(self.root, "tracking_results.json")

    @property
    def asd_results_json(self) -> str:
        """asd_results.json 路径"""
        return os.path.join(self.root, "asd_results.json")

    @property
    def process_status_json(self) -> str:
        """results/process_status.json 路径"""
        return os.path.join(self.results, "process_status.json")

    @property
    def all_tracks_mp4(self) -> str:
        """pytracks/all_tracks.mp4 路径"""
        return os.path.join(self.pytracks, "all_tracks.mp4")

    @property
    def all_tracks_with_dialog_mp4(self) -> str:
        """pytracks/all_tracks_with_dialog.mp4 路径"""
        return os.path.join(self.pytracks, "all_tracks_with_dialog.mp4")

    @property
    def person_subtitle_mapping_json(self) -> str:
        """results/person_subtitle_mapping.json 路径"""
        return os.path.join(self.results, "person_subtitle_mapping.json")

    def init_dirs(self) -> None:
        """创建所有需要的子目录"""
        for d in [
            self.pyavi,
            self.pyframes,
            self.pycrop,
            self.pytracks,
            self.results,
            self.portraits,
        ]:
            os.makedirs(d, exist_ok=True)

    def cleanup_frames(self) -> None:
        """清理帧截图目录，释放磁盘空间"""
        shutil.rmtree(self.pyframes, ignore_errors=True)

    def cleanup_crop(self) -> None:
        """清理 crop 中间文件（预测完成后可调用）"""
        shutil.rmtree(self.pycrop, ignore_errors=True)

    def cleanup_frames_and_crop(self) -> None:
        """同时清理 frames 与 crop"""
        self.cleanup_frames()
        self.cleanup_crop()
