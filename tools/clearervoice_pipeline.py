"""
基于 ClearVoice 的视频音频语音增强工具

从 MP4 中提取音频，使用 ClearVoice 模型进行语音增强，
再将增强后的音频合并回视频，支持单文件和批量处理。
"""

import logging
import os
import subprocess
import tempfile
from typing import Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_FORMATS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm"}


class ClearVoicePipeline:
    """
    使用 ClearVoice 对视频音频进行语音增强。

    模型在初始化时一次性加载，后续调用 enhance / enhance_many 直接推理，
    避免重复加载开销。
    """

    def __init__(
        self,
        task: str = "speech_enhancement",
        model_name: str = "MossFormer2_SE_48K",
        device: str = "cuda:0",
    ):
        """
        Args:
            task: ClearVoice 任务类型 (speech_enhancement / speech_separation / target_speaker_extraction)
            model_name: 模型名称，如 MossFormer2_SE_48K, MossFormerGAN_SE_16K 等
            device: 指定 CUDA 设备，避免多卡环境下张量设备不一致导致 RuntimeError
        """
        self.task = task
        self.model_name = model_name
        self.device = device

        # 在加载模型前显式设置 CUDA 设备，确保模型权重与推理时输入在同一设备
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(device)

        logger.info(f"正在加载 ClearVoice 模型: task={task}, model={model_name}, device={device}")
        from clearvoice import ClearVoice

        self.model = ClearVoice(task=task, model_names=[model_name])

        # ClearVoice 内部用 get_free_gpu() 选显存最多的卡，可能与多线程下的默认设备不一致，
        # 导致 weight on cuda:1 / input on cuda:0。显式将模型和 device 统一到指定设备。
        if device.startswith("cuda") and torch.cuda.is_available():
            target = torch.device(device)
            for m in self.model.models:
                if hasattr(m, "model") and m.model is not None:
                    m.model.to(target)
                m.device = target

        logger.info("ClearVoice 模型加载完成")

    @staticmethod
    def _extract_audio(video_path: str, wav_path: str) -> str:
        """从视频中提取音频为 WAV"""
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            wav_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 提取音频失败: {result.stderr}")
        return wav_path

    @staticmethod
    def _replace_audio(video_path: str, enhanced_wav: str, output_path: str) -> str:
        """用增强后的音频替换原视频中的音频轨"""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", enhanced_wav,
            "-c:v", "copy",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 合并音频失败: {result.stderr}")
        return output_path

    def enhance(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        对单个 MP4 视频的音频进行语音增强。

        Args:
            input_path:  输入视频路径
            output_path: 输出视频路径（默认: <input_stem>_enhanced.<ext>）

        Returns:
            输出视频路径
        """
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_enhanced{ext}"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_wav = os.path.join(tmpdir, "raw_audio.wav")
            enhanced_wav = os.path.join(tmpdir, "enhanced_audio.wav")

            logger.info(f"[1/3] 提取音频: {input_path}")
            self._extract_audio(input_path, raw_wav)

            logger.info(f"[2/3] ClearVoice 语音增强 ({self.model_name})")
            # 推理前设置默认设备，确保 ClearVoice 内部创建的 tensor 与模型在同一设备
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.set_device(int(self.device.split(":")[-1]))
            result_wav = self.model(input_path=raw_wav, online_write=False)
            self.model.write(result_wav, output_path=enhanced_wav)

            logger.info(f"[3/3] 合并增强音频 -> {output_path}")
            self._replace_audio(input_path, enhanced_wav, output_path)

        logger.info(f"增强完成: {output_path}")
        return output_path

    def enhance_many(
        self,
        input_paths: list[str],
        output_dir: str,
    ) -> list[str]:
        """
        批量增强多个视频的音频。

        Args:
            input_paths: 输入视频路径列表
            output_dir:  输出目录

        Returns:
            输出文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        outputs = []
        for path in input_paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            ext = os.path.splitext(path)[1]
            out_path = os.path.join(output_dir, f"{stem}_enhanced{ext}")
            self.enhance(path, out_path)
            outputs.append(out_path)
        return outputs


def build_clearvoice_model(
    task: str = "speech_enhancement",
    model_name: str = "MossFormer2_SE_48K",
) -> ClearVoicePipeline:
    """构建并返回 ClearVoicePipeline 实例（便于外部复用）。"""
    return ClearVoicePipeline(task=task, model_name=model_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ClearVoice 视频音频语音增强工具"
    )
    parser.add_argument("input", type=str, help="输入视频路径")
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出视频路径（默认: <input_stem>_enhanced.<ext>）",
    )
    parser.add_argument(
        "--task", type=str, default="speech_enhancement",
        help="ClearVoice 任务类型",
    )
    parser.add_argument(
        "--model", type=str, default="MossFormer2_SE_48K",
        help="模型名称",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="CUDA 设备 (如 cuda:0)，避免多卡环境下设备不一致",
    )

    args = parser.parse_args()

    pipeline = ClearVoicePipeline(task=args.task, model_name=args.model, device=args.device)
    pipeline.enhance(args.input, args.output)
