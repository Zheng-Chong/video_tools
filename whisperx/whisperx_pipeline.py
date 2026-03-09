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
from typing import Dict, List, Optional
import torch
import whisperx

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MovieTranscriber:
    """
    基于 WhisperX 的高精度电影字幕与说话人分离工具
    """
    
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.webm'}
    
    def __init__(
        self,
        hf_token: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        compute_type: str = "float16" # 如果显存极小(如小于8G)，可改为 "int8"
    ):
        """
        初始化转录器
        
        Args:
            hf_token: HuggingFace Token，用于下载 pyannote 模型
            device: 运行设备 ("cuda" 或 "cpu")
            compute_type: 计算精度 ("float16", "int8", "float32")
        """
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            raise ValueError("必须提供 HuggingFace Token 用于加载 Pyannote 模型")
            
        self.device = device
        self.compute_type = compute_type if device == "cuda" else "float32"
        self._temp_files: List[str] = []

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
        whisper_model: str = "large-v3",
        language: Optional[str] = None,
        batch_size: int = 16,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Dict:
        """
        执行完整的分离和转录流水线
        """
        audio_path = input_path
        if os.path.splitext(input_path)[1].lower() in self.SUPPORTED_VIDEO_FORMATS:
            audio_path = self._extract_audio(input_path)

        try:
            logger.info("加载音频至内存...")
            audio = whisperx.load_audio(audio_path)

            # ==========================================
            # 阶段 1: VAD + Whisper 批量转录
            # ==========================================
            logger.info(f"--> [阶段 1/3] 加载 Whisper 模型 ({whisper_model}) 并执行转录")
            model = whisperx.load_model(
                whisper_model, 
                self.device, 
                compute_type=self.compute_type, 
                language=language
            )
            
            # 这里是提速的核心：由于有 VAD 预处理，可以直接 batch 并发推理
            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]
            logger.info(f"转录完成，检测到语言: {detected_language}")
            
            # 及时释放 Whisper 大模型显存
            del model
            self._cleanup_memory()

            # ==========================================
            # 阶段 2: 词级时间戳强制对齐 (Forced Alignment)
            # ==========================================
            logger.info("--> [阶段 2/3] 加载 Wav2vec2 模型并执行词级对齐")
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_language, 
                device=self.device
            )
            
            result = whisperx.align(
                result["segments"], 
                align_model, 
                metadata, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            
            # 释放对齐模型显存
            del align_model
            self._cleanup_memory()

            # ==========================================
            # 阶段 3: 说话人日记与词级映射
            # ==========================================
            logger.info("--> [阶段 3/3] 加载 Pyannote 模型并执行说话人分离")
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self.hf_token, 
                device=self.device
            )
            
            # 执行 Diarization
            diarize_segments = diarize_model(
                audio, 
                min_speakers=min_speakers, 
                max_speakers=max_speakers
            )
            
            # 将 Diarization 结果分配给刚才对齐好的单词
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # 释放 Pyannote 模型显存
            del diarize_model
            self._cleanup_memory()
            
            logger.info("全流程处理完毕！")
            return result

        finally:
            # 清理临时提取的音频文件
            for f in self._temp_files:
                if os.path.exists(f):
                    os.unlink(f)
            self._temp_files.clear()

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
    
    # 执行处理
    result = transcriber.process(
        input_path=args.input,
        batch_size=args.batch_size
    )
    
    # 保存结果
    out_prefix = args.output or os.path.splitext(args.input)[0]
    
    # 保存 JSON (包含所有精确到词级别的时间戳信息，便于二次开发)
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    # 保存 SRT 字幕
    transcriber.save_srt(result, f"{out_prefix}.srt")