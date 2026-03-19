"""
Movie Transcriber based on WhisperX

结合 VAD、Faster-Whisper、Wav2vec2词级对齐与 Pyannote 说话人日记，
实现高精度、极速、低显存的电影级别说话人+字幕分离。
"""

import gc
import json
import logging
import os
import subprocess
import tempfile
import warnings
from typing import Dict, List, Optional
import torch
import whisperx

# 抑制无语音片段产生的警告
for _name in ("whisperx.vads", "whisperx.vads.pyannote", "pyannote", "pyannote.audio"):
    logging.getLogger(_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MovieTranscriber:
    """
    基于 WhisperX 的高精度电影字幕与说话人分离工具。
    所有模型在初始化时一次性加载，后续调用 process/process_many 直接推理。
    """
    
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.webm'}
    
    def __init__(
        self,
        hf_token: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        compute_type: str = "float16",
        whisper_model: str = "large-v3",
        language: Optional[str] = "en",
    ):
        """
        初始化转录器，预加载 Whisper、Wav2vec2 对齐、Pyannote 说话人分离三个模型。

        Args:
            hf_token: HuggingFace Token，用于下载 pyannote 模型
            device: 运行设备 ("cuda" 或 "cpu")
            compute_type: 计算精度 ("float16", "int8", "float32")
            whisper_model: Whisper 模型名称，如 "large-v3"
            language: 预加载的对齐模型语言，None 表示不预加载（首次推理时按检测语言加载）
        """
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            raise ValueError("必须提供 HuggingFace Token 用于加载 Pyannote 模型")
            
        self.device = device
        self.compute_type = compute_type if device == "cuda" else "float32"
        self._temp_files: List[str] = []

        logger.info(f"[初始化 1/3] 加载 Whisper 模型 ({whisper_model})...")
        self._whisper_model = whisperx.load_model(
            whisper_model,
            self.device,
            compute_type=self.compute_type,
            language=language,
        )

        self._align_models: Dict[str, tuple] = {}
        if language:
            logger.info(f"[初始化 2/3] 加载 Wav2vec2 对齐模型 ({language})...")
            align_model, metadata = whisperx.load_align_model(
                language_code=language, device=self.device
            )
            self._align_models[language] = (align_model, metadata)
        else:
            logger.info("[初始化 2/3] 对齐模型将在首次推理时按检测语言加载")

        logger.info("[初始化 3/3] 加载 Pyannote 说话人分离模型...")
        self._diarize_model = whisperx.diarize.DiarizationPipeline(
            token=self.hf_token, device=self.device
        )

        logger.info("MovieTranscriber 所有模型加载完成")

    def _get_align_model(self, language: str) -> tuple:
        """获取指定语言的对齐模型，首次使用时加载并缓存"""
        if language not in self._align_models:
            logger.info(f"加载 Wav2vec2 对齐模型 ({language})...")
            align_model, metadata = whisperx.load_align_model(
                language_code=language, device=self.device
            )
            self._align_models[language] = (align_model, metadata)
        return self._align_models[language]

    def _extract_audio(self, video_path: str) -> str:
        """统一提取一次音频，供全流程使用"""
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        output_path = temp_audio.name
        temp_audio.close()
        self._temp_files.append(output_path)
        
        logger.info(f"正在从视频提取音频 (16kHz, 单声道): {video_path}")
        cmd = [
            'ffmpeg', '-i', video_path, '-vn',
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-y', output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    def _cleanup_memory(self):
        """强制清理显存和内存（关键优化点）"""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def process(
        self,
        input_path: str,
        language: Optional[str] = "en",
        batch_size: int = 16,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Dict:
        """
        使用预加载的模型执行完整的转录+对齐+说话人分离流水线。
        """
        audio_path = input_path
        if os.path.splitext(input_path)[1].lower() in self.SUPPORTED_VIDEO_FORMATS:
            audio_path = self._extract_audio(input_path)

        try:
            audio = whisperx.load_audio(audio_path)

            result = self._whisper_model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]

            align_model, metadata = self._get_align_model(detected_language)
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )

            diarize_segments = self._diarize_model(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)

            return result

        finally:
            for f in self._temp_files:
                if os.path.exists(f):
                    os.unlink(f)
            self._temp_files.clear()

    def process_many(
        self,
        input_paths: List[str],
        language: Optional[str] = "en",
        batch_size: int = 16,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[Dict]:
        """
        批量处理多个片段，直接使用预加载的模型，无需重复加载/销毁。
        """
        if not input_paths:
            return []

        results: List[Dict] = []

        for input_path in input_paths:
            audio_path = input_path
            if os.path.splitext(input_path)[1].lower() in self.SUPPORTED_VIDEO_FORMATS:
                audio_path = self._extract_audio(input_path)

            try:
                audio = whisperx.load_audio(audio_path)

                result = self._whisper_model.transcribe(audio, batch_size=batch_size)
                detected_language = result["language"]

                align_model, metadata = self._get_align_model(detected_language)
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )

                diarize_segments = self._diarize_model(
                    audio,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)

                results.append(result)
            finally:
                for f in self._temp_files:
                    if os.path.exists(f):
                        os.unlink(f)
                self._temp_files.clear()

        return results

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """将秒数转换为 SRT 时间戳格式 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def save_srt(self, result: Dict, output_path: str):
        """将 WhisperX 的结果保存为带说话人的 SRT"""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        segments = result.get("segments", [])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, start=1):
                start_ts = self.format_timestamp(seg["start"])
                end_ts = self.format_timestamp(seg["end"])
                speaker = seg.get("speaker", "UNKNOWN")
                text = seg["text"].strip()
                
                f.write(f"{i}\n")
                f.write(f"{start_ts} --> {end_ts}\n")
                f.write(f"[{speaker}] {text}\n\n")
                
        logger.info(f"SRT 字幕已保存至: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='高精度电影语音转文字工具 (WhisperX)')
    parser.add_argument('input', type=str, help='视频或音频路径')
    parser.add_argument('--output', type=str, default=None, help='输出路径前缀')
    parser.add_argument('--hf_token', type=str, required=True, help='HuggingFace Token')
    parser.add_argument('--batch_size', type=int, default=16, help='Whisper 推理批次大小')
    
    args = parser.parse_args()
    
    transcriber = MovieTranscriber(hf_token=args.hf_token)
    
    result = transcriber.process(
        input_path=args.input,
        batch_size=args.batch_size,
    )
    
    out_prefix = args.output or os.path.splitext(args.input)[0]
    
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    transcriber.save_srt(result, f"{out_prefix}.srt")