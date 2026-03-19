"""
基于 SAM-Audio 的视频人声提取工具

利用 Meta SAM-Audio 模型的文本提示（Text Prompting）能力，
从视频音频中分离出干净的人声，支持单文件和批量处理。
"""

import gc
import logging
import os
import subprocess
import tempfile
from typing import List, Optional

import torch
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Models", "sam-audio-base",
)

VOICE_DESCRIPTIONS = [
    "A person speaking clearly",
    "Human speech and voice",
    "A man or woman talking",
]

SUPPORTED_VIDEO_FORMATS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm"}


class SAMAudioVoiceExtractor:
    """
    使用 SAM-Audio 从视频/音频中提取干净人声。

    模型在初始化时一次性加载，后续调用 extract / extract_many 直接推理，
    避免重复加载开销。
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
        description: str = "A person speaking clearly",
    ):
        """
        Args:
            model_path: SAM-Audio 模型路径（本地目录或 HuggingFace ID）
            device: 运行设备，默认自动检测
            description: 用于提示模型的文本描述（描述目标人声）
        """
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.description = description
        self._temp_files: List[str] = []

        logger.info(f"正在加载 SAM-Audio 模型: {model_path}")
        from sam_audio import SAMAudio, SAMAudioProcessor

        self.model = SAMAudio.from_pretrained(model_path).to(self.device).eval()
        self.processor = SAMAudioProcessor.from_pretrained(model_path)
        self.sample_rate = self.processor.audio_sampling_rate

        logger.info(
            f"SAM-Audio 加载完成 (device={self.device}, sr={self.sample_rate})"
        )

    def _extract_audio_from_video(self, video_path: str) -> str:
        """从视频中提取音频为临时 WAV 文件（采样率与模型匹配）"""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = tmp.name
        tmp.close()
        self._temp_files.append(wav_path)

        logger.info(f"从视频提取音频 ({self.sample_rate}Hz): {video_path}")
        cmd = [
            "ffmpeg", "-i", video_path, "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-y", wav_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 提取音频失败: {result.stderr}")
        return wav_path

    def _cleanup(self):
        """清理临时文件和显存"""
        for f in self._temp_files:
            if os.path.exists(f):
                os.unlink(f)
        self._temp_files.clear()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def extract(
        self,
        input_path: str,
        output_path: str,
        description: Optional[str] = None,
        anchors: Optional[List] = None,
    ) -> str:
        """
        从单个视频/音频文件中提取干净人声。

        Args:
            input_path:  输入视频或音频路径
            output_path: 输出干净人声 WAV 路径
            description: 文本描述（覆盖默认值），如 "A woman speaking"
            anchors:     时间锚点列表 [["+/-", start, end], ...]，可选

        Returns:
            output_path
        """
        desc = description or self.description
        ext = os.path.splitext(input_path)[1].lower()

        audio_path = input_path
        if ext in SUPPORTED_VIDEO_FORMATS:
            audio_path = self._extract_audio_from_video(input_path)

        try:
            logger.info(f"SAM-Audio 分离人声 (prompt={desc!r}): {input_path}")

            kwargs = dict(
                audios=[audio_path],
                descriptions=[desc],
            )
            if anchors:
                kwargs["anchors"] = [anchors]

            inputs = self.processor(**kwargs).to(self.device)

            with torch.inference_mode():
                result = self.model.separate(inputs)

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torchaudio.save(
                output_path,
                result.target[0].unsqueeze(0).cpu(),
                self.sample_rate,
            )
            logger.info(f"干净人声已保存: {output_path}")
            return output_path

        finally:
            self._cleanup()

    def extract_many(
        self,
        input_paths: List[str],
        output_dir: str,
        description: Optional[str] = None,
    ) -> List[str]:
        """
        批量提取多个文件的人声。

        Args:
            input_paths: 输入文件路径列表
            output_dir:  输出目录
            description: 文本描述（覆盖默认值）

        Returns:
            输出文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        outputs = []
        for path in input_paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(output_dir, f"{stem}_voice.wav")
            self.extract(path, out_path, description=description)
            outputs.append(out_path)
        return outputs


def build_sam_audio_model(
    model_path: str = DEFAULT_MODEL_PATH,
    device: Optional[str] = None,
    description: str = "A person speaking clearly",
) -> SAMAudioVoiceExtractor:
    """构建并返回 SAMAudioVoiceExtractor 实例（便于外部复用）。"""
    return SAMAudioVoiceExtractor(
        model_path=model_path,
        device=device,
        description=description,
    )


def run_sam_audio_clip(
    input_path: str,
    output_path: str,
    description: str = "A person speaking clearly",
    model: Optional[SAMAudioVoiceExtractor] = None,
) -> str:
    """
    单次处理：从视频/音频中提取干净人声。

    Args:
        input_path:  输入视频或音频
        output_path: 输出干净人声 WAV
        description: 文本提示
        model:       可传入已初始化的模型实例以复用

    Returns:
        output_path
    """
    extractor = model if model is not None else build_sam_audio_model()
    return extractor.extract(input_path, output_path, description=description)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SAM-Audio 视频人声提取工具"
    )
    parser.add_argument("input", type=str, help="输入视频或音频路径")
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出 WAV 路径（默认: <input_stem>_voice.wav）",
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_PATH,
        help="SAM-Audio 模型路径",
    )
    parser.add_argument(
        "--description", type=str, default="A person speaking clearly",
        help="文本提示，描述要提取的声音",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="运行设备 (cuda / cpu)",
    )

    args = parser.parse_args()

    out = args.output or os.path.splitext(args.input)[0] + "_voice.wav"

    extractor = SAMAudioVoiceExtractor(
        model_path=args.model_path,
        device=args.device,
        description=args.description,
    )
    extractor.extract(args.input, out)
