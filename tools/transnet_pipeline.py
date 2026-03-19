"""
TransNetV2 视频镜头检测模块
=============================

本模块提供基于 TransNetV2 深度学习模型的视频镜头边界检测功能。
TransNetV2 是一个用于自动检测视频中场景/镜头切换点的神经网络模型。

主要功能:
    - 自动检测视频中的镜头边界
    - 将视频按镜头分割并保存为独立文件
    - 支持音频同步提取
    - 支持断点续传处理
    - 支持自定义裁剪尺寸

典型使用场景:
    - 视频编辑预处理
    - 视频内容分析
    - 数据集构建与标注
    - 视频摘要生成

使用示例
--------

基础用法 - 检测视频镜头并分割保存:

    >>> from transnet import TransNetV2Pipeline
    >>> 
    >>> # 初始化管道（自动下载模型或使用本地模型）
    >>> pipeline = TransNetV2Pipeline()
    >>> 
    >>> # 读取视频文件
    >>> pipeline.read_video("input_video.mp4") 
    >>> 
    >>> # 执行镜头检测并保存分割结果
    >>> shot_count = pipeline.shot_detect(
    ...     output_dir="./output",
    ...     threshold=0.2,      # 检测阈值，越小越敏感
    ...     min_frames=0       # 0=保留所有镜头，>0 则过滤短于该帧数的镜头
    ... )
    >>> print(f"检测到 {shot_count} 个镜头")

指定 GPU 设备:

    >>> pipeline = TransNetV2Pipeline(device="cuda:0")

使用本地模型权重:

    >>> pipeline = TransNetV2Pipeline(model_path="/path/to/model_weights/")

带裁剪的镜头检测（去除黑边）:

    >>> shot_count = pipeline.shot_detect(
    ...     output_dir="./output",
    ...     target_width=1920,
    ...     target_height=1080
    ... )

依赖项
------
- torch >= 1.9.0
- torchvision >= 0.10.0
- torchaudio >= 0.9.0
- huggingface_hub
- tqdm
- numpy

作者: TransNetV2 PyTorch Implementation
许可证: MIT License
"""

import os
import warnings
import glob
from fractions import Fraction
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import random
import torchaudio
from huggingface_hub import hf_hub_download
from torchvision import transforms
from torchvision.io import VideoReader, write_video
from tqdm import tqdm


__all__ = ["TransNetV2", "TransNetV2Pipeline"]


_VIDEO_CODEC_CANDIDATES = ("libx264", "h264", "mpeg4")
_AUDIO_CODEC_CANDIDATES = ("aac", "mp3")


class TransNetV2Pipeline:
    """
    TransNetV2 视频镜头检测管道类。

    该类封装了 TransNetV2 模型的完整推理流程，提供从视频读取、
    镜头边界检测到视频分割保存的端到端处理能力。

    Attributes:
        device (torch.device): 模型运行的计算设备 (CPU/CUDA)。
        model (TransNetV2): 加载的 TransNetV2 神经网络模型。
        reader (VideoReader): 当前视频的读取器实例。
        video_path (str): 当前加载的视频文件路径。
        audio_waveform (torch.Tensor): 预加载的音频波形数据。
        audio_sample_rate (int): 音频采样率。

    Example:
        >>> # 完整的镜头检测流程
        >>> pipeline = TransNetV2Pipeline(device="cuda")
        >>> pipeline.read_video("movie.mp4")
        >>> num_shots = pipeline.shot_detect("./shots_output")
        >>> print(f"成功分割 {num_shots} 个镜头")
    """

    def __init__(self, model_path: str = "Sn4kehead/TransNetV2", device: str = None):
        """
        初始化 TransNetV2 推理管道。

        Args:
            model_path (str, optional): 模型权重路径。可以是以下两种形式:
                - HuggingFace Hub 仓库 ID (如 "Sn4kehead/TransNetV2")
                - 本地目录路径 (包含 transnetv2-pytorch-weights.pth 文件)
                默认值: "Sn4kehead/TransNetV2"
            device (str, optional): 计算设备。可选值:
                - "cuda" / "cuda:0" / "cuda:1" 等 (GPU)
                - "cpu" (CPU)
                - None (自动选择，优先使用 GPU)
                默认值: None

        Raises:
            FileNotFoundError: 当指定的本地模型路径不存在时。
            RuntimeError: 当模型加载失败时。

        Example:
            >>> # 使用默认设置（自动下载模型，自动选择设备）
            >>> pipeline = TransNetV2Pipeline()
            >>>
            >>> # 指定 GPU 和本地模型
            >>> pipeline = TransNetV2Pipeline(
            ...     model_path="/models/transnet/",
            ...     device="cuda:0"
            ... )
        """
        # 自动检测可用设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 加载模型权重
        if os.path.exists(model_path):
            # 从本地路径加载
            state_dict = torch.load(
                os.path.join(model_path, "transnetv2-pytorch-weights.pth"),
                map_location=self.device,
            )
        else:
            # 从 HuggingFace Hub 下载
            state_dict = hf_hub_download(
                repo_id=model_path, filename="transnetv2-pytorch-weights.pth"
            )
            state_dict = torch.load(state_dict, map_location=self.device)

        # 初始化并加载模型
        self.model = TransNetV2()
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)

        # 初始化视频/音频相关属性
        self.reader = None
        self.video_path = None
        self.audio_waveform = None
        self.audio_sample_rate = None

    def read_video(self, video_path: str) -> None:
        """
        读取并加载视频文件。

        该方法会同时加载视频流和音频流（如果存在）。音频流会被
        完整预加载到内存中，以便后续高效的音频片段提取。

        Args:
            video_path (str): 视频文件路径，必须是 MP4 格式。

        Raises:
            AssertionError: 当视频不是 MP4 格式时。
            RuntimeError: 当视频文件无法读取时。

        Warning:
            如果视频不包含音频流或音频读取失败，会发出警告但不会
            中断处理流程，后续保存的视频将不包含音频。

        Example:
            >>> pipeline = TransNetV2Pipeline()
            >>> pipeline.read_video("/data/videos/sample.mp4")
        """
        assert video_path.endswith(".mp4"), "仅支持 MP4 格式视频文件"

        self.video_path = video_path
        self.reader = VideoReader(video_path, "video")

        # 尝试加载音频流
        try:
            self.audio_waveform, self.audio_sample_rate = torchaudio.load(video_path)
        except Exception as e:
            warnings.warn(f"无法加载音频流: {e}。将继续处理但不包含音频。")
            self.audio_waveform = None
            self.audio_sample_rate = None

    def get_video_frames(self, num_frames: int, keep_high_res: bool = True) -> tuple:
        """
        从当前视频位置读取指定数量的帧。

        Args:
            num_frames (int): 要读取的帧数。
            keep_high_res (bool): 是否保留并返回原始高清帧。如果仅用于推理可设为 False 节省内存。
        """
        frames = []
        for _ in range(num_frames):
            try:
                frames.append(next(self.reader)["data"])
            except StopIteration:
                break

        if len(frames) == 0:
            raise ValueError("无法从视频中读取帧数据")

        frames_tensor = torch.stack(frames)

        # 缩放到模型输入尺寸 (27x48)
        resized_frames = transforms.functional.resize(
            frames_tensor,
            size=[27, 48],
            interpolation=transforms.InterpolationMode.BILINEAR,
        )

        resized_frames = resized_frames.permute(0, 2, 3, 1).unsqueeze(0)

        # 如果不需要保存原视频，直接返回 None 释放原图内存
        if keep_high_res:
            frames_tensor = frames_tensor.permute(0, 2, 3, 1).unsqueeze(0)
            return frames_tensor, resized_frames
        else:
            return None, resized_frames

    def _extract_audio_segment_torch(
        self, start_frame: int, end_frame: int, video_fps: float
    ) -> tuple:
        """
        根据帧索引从预加载的音频中提取对应片段。

        该方法使用纯张量操作，执行效率高。

        Args:
            start_frame (int): 起始帧索引。
            end_frame (int): 结束帧索引。
            video_fps (float): 视频帧率。

        Returns:
            tuple: 包含两个元素:
                - audio_segment (torch.Tensor or None): 音频片段张量，形状为 [C, S]
                - sample_rate (int or None): 音频采样率
                如果没有可用音频或范围无效，返回 (None, None)
        """
        if self.audio_waveform is None:
            return None, None

        # 计算对应的音频样本索引
        start_sample = int((start_frame / video_fps) * self.audio_sample_rate)
        end_sample = int((end_frame / video_fps) * self.audio_sample_rate)

        # 确保索引在有效范围内
        start_sample = max(0, start_sample)
        end_sample = min(self.audio_waveform.shape[1], end_sample)

        if start_sample >= end_sample:
            return None, None

        # 直接切片获取音频片段
        audio_segment = self.audio_waveform[:, start_sample:end_sample]

        return audio_segment, self.audio_sample_rate

    def _save_video(
        self,
        frames_tensor: torch.Tensor,
        output_filename: str,
        start_frame: int,
        end_frame: int,
        video_fps: float,
    ) -> None:
        """
        将帧序列保存为视频文件。

        该方法会自动检测并包含对应的音频片段（如果可用）。

        Args:
            frames_tensor (torch.Tensor): 视频帧张量，形状为 [N, H, W, C]。
            output_filename (str): 输出视频文件路径。
            start_frame (int): 起始帧索引（用于音频对齐）。
            end_frame (int): 结束帧索引（用于音频对齐）。
            video_fps (float): 视频帧率。

        Note:
            - 自动按优先级尝试可用视频编码器（libx264/h264/mpeg4）
            - 自动按优先级尝试可用音频编码器（aac/mp3）
        """
        video_fps = Fraction(video_fps).limit_denominator(1000)
        contiguous_frames = frames_tensor.contiguous()

        # 提取对应的音频片段
        audio_segment, audio_fps = self._extract_audio_segment_torch(
            start_frame, end_frame, float(video_fps)
        )

        audio_kwargs = None
        if audio_segment is not None and audio_segment.shape[1] > 0:
            audio_kwargs = {
                "audio_array": audio_segment.contiguous(),
                "audio_fps": int(audio_fps),
            }

        self._write_video_with_fallback(
            output_filename=output_filename,
            contiguous_frames=contiguous_frames,
            video_fps=video_fps,
            audio_kwargs=audio_kwargs,
        )

    def _write_video_with_fallback(
        self,
        output_filename: str,
        contiguous_frames: torch.Tensor,
        video_fps: Fraction,
        audio_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        尝试可用编码器写入视频，避免环境缺少特定编解码器时直接失败。
        """
        last_error = None

        for video_codec in _VIDEO_CODEC_CANDIDATES:
            if audio_kwargs:
                for audio_codec in _AUDIO_CODEC_CANDIDATES:
                    try:
                        write_video(
                            filename=output_filename,
                            video_array=contiguous_frames,
                            fps=video_fps,
                            video_codec=video_codec,
                            audio_codec=audio_codec,
                            **audio_kwargs,
                        )
                        return
                    except Exception as exc:
                        last_error = exc
            else:
                try:
                    write_video(
                        filename=output_filename,
                        video_array=contiguous_frames,
                        fps=video_fps,
                        video_codec=video_codec,
                    )
                    return
                except Exception as exc:
                    last_error = exc

        # 音视频都失败时，最后再尝试一轮“仅视频导出”，避免因音频编码器不可用导致整体失败。
        if audio_kwargs:
            for video_codec in _VIDEO_CODEC_CANDIDATES:
                try:
                    write_video(
                        filename=output_filename,
                        video_array=contiguous_frames,
                        fps=video_fps,
                        video_codec=video_codec,
                    )
                    warnings.warn(
                        f"导出 {output_filename} 时音频编码失败，已降级为无音频输出。",
                        RuntimeWarning,
                    )
                    return
                except Exception as exc:
                    last_error = exc

        raise RuntimeError(
            "写入视频失败：未找到可用编码器。"
            f" 已尝试视频编码器={list(_VIDEO_CODEC_CANDIDATES)}，"
            f" 音频编码器={list(_AUDIO_CODEC_CANDIDATES)}。"
        ) from last_error

    def segment_shot_detect(self, frames: torch.Tensor, threshold: float = 0.2):
        """
        对一段帧序列执行镜头边界检测。

        Args:
            frames (torch.Tensor): 缩放后的帧张量，形状为 [1, N, 27, 48, C]。
            threshold (float, optional): 检测阈值，范围 [0, 1]。
                值越小检测越敏感，越容易检测到镜头边界。
                默认值: 0.2

        Returns:
            np.ndarray: 场景边界数组，形状为 [M, 2]，
                每行 [start, end] 表示一个场景的起止帧索引（相对于输入帧序列）。
        """
        with torch.no_grad():
            frames_gpu = frames.to(self.device)
            predictions, _ = self.model(frames_gpu)
            predictions = torch.sigmoid(predictions).cpu().numpy()[0, :, 0]

        scenes = TransNetV2.predictions_to_scenes(predictions, threshold=threshold)
        return scenes

    def extract_shots_from_list(
        self,
        list_file: str,
        output_dir: str,
        target_width: int = None,
        target_height: int = None,
    ):
        """
        根据给定的镜头列表文件，从当前视频中提取并保存镜头片段。

        Args:
            list_file (str): 包含镜头边界的文本文件路径 (如 shots_list.txt)，
                             每行格式为 "start_frame end_frame"。
            output_dir (str): 保存提取出的 MP4 文件的目标目录。
            target_width (int, optional): 目标裁剪宽度。
            target_height (int, optional): 目标裁剪高度。

        Raises:
            ValueError: 未加载视频或列表文件不存在时抛出。
        """
        if self.reader is None or self.video_path is None:
            raise ValueError("请先调用 `TransNetV2Pipeline.read_video()` 加载视频")

        if not os.path.exists(list_file):
            raise FileNotFoundError(f"找不到指定的列表文件: {list_file}")

        # 读取并解析列表文件（跳过 # 开头的注释行，如 # COMPLETE）
        shots = []
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        shots.append((int(parts[0]), int(parts[1])))
                    except ValueError:
                        pass  # 跳过注释行

        if not shots:
            print("列表文件为空，无镜头需要提取。")
            return 0

        # 获取视频元数据
        metadata = self.reader.get_metadata()
        video_fps = metadata["video"]["fps"][0]

        os.makedirs(output_dir, exist_ok=True)

        print(
            f"准备从 {os.path.basename(self.video_path)} 中提取 {len(shots)} 个镜头..."
        )

        extracted_count = 0
        black_bar_skipped_count = 0

        for i, (start_frame, end_frame) in enumerate(tqdm(shots, desc="提取镜头")):
            # 计算起始时间戳 (秒)
            start_time_sec = start_frame / video_fps

            # 使用 seek 跳转到最近的关键帧位置
            self.reader.seek(start_time_sec)

            num_frames_to_read = end_frame - start_frame
            frames = []

            # 读取该镜头的帧
            for _ in range(num_frames_to_read):
                try:
                    frames.append(next(self.reader)["data"])
                except StopIteration:
                    break

            if not frames:
                continue

            frames_tensor = torch.stack(frames)
            # 调整维度顺序: [N, C, H, W] -> [N, H, W, C]
            frames_tensor = frames_tensor.permute(0, 2, 3, 1)

            # 可选：裁剪尺寸
            if target_width is not None and target_height is not None:
                # 假设你原本类里有这个方法，如果没有需要自己实现或去掉这块逻辑
                cropped_frames = self._crop_frames_to_target_size(
                    frames_tensor, target_width, target_height
                )
                if cropped_frames is None:
                    black_bar_skipped_count += 1
                    continue
                frames_tensor = cropped_frames

            # 构造输出文件名
            output_filename = os.path.join(
                output_dir, f"scene_{i+1:03d}_{start_frame:06d}_{end_frame:06d}.mp4"
            )

            # 调用原有的保存逻辑（包含音频提取和写入）
            self._save_video(
                frames_tensor, output_filename, start_frame, end_frame, video_fps
            )
            extracted_count += 1

        print("\n提取完成！")
        print(f"成功提取: {extracted_count} 个镜头")
        if black_bar_skipped_count > 0:
            print(f"因黑边跳过: {black_bar_skipped_count} 个镜头")

        return extracted_count

    def shot_detect(
        self,
        output_dir: str,
        segment_frames: int = 1000,
        threshold: float = 0.2,
        min_frames: int = 0,
        target_width: int = None,
        target_height: int = None,
        save_mp4: bool = True,  # <--- 新增参数：是否切分并保存 MP4
    ) -> int:
        """
        执行完整的视频镜头检测与分割流程。
        ... (此处省略部分注释，可保留你原本的注释并加上 save_mp4 的说明) ...
        """
        if self.reader is None:
            raise ValueError("请先调用 `TransNetV2Pipeline.read_video()` 方法加载视频")

        # 获取视频元数据
        metadata = self.reader.get_metadata()
        video_fps = metadata["video"]["fps"][0]
        total_frames = int(metadata["video"]["duration"][0] * video_fps)

        # 创建输出目录
        output_file = os.path.join(output_dir, "shots_list.txt")
        os.makedirs(output_dir, exist_ok=True)  # 确保根目录存在

        output_video_dir = None
        if save_mp4:
            output_video_dir = os.path.join(output_dir, "shots")
            os.makedirs(output_video_dir, exist_ok=True)

        # 检查是否已全部检测完成（shots_list.txt 含结束符则直接跳过）
        SHOTS_LIST_COMPLETE_MARKER = "# COMPLETE"
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                lines = [line for line in f.read().splitlines() if line]
            if lines and lines[-1].strip() == SHOTS_LIST_COMPLETE_MARKER:
                data_lines = [
                    l for l in lines
                    if not l.strip().startswith("#")
                ]
                scene_count = len(data_lines)
                print(f"检测到结束符，跳过已完成的电影（共 {scene_count} 个镜头）")
                return scene_count

        # 初始化进度条
        progress_bar = tqdm(total=total_frames, desc=os.path.dirname(self.video_path))

        # 断点续传：跳过已处理的帧
        last_end = 0
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                lines = [
                    l for l in f.read().splitlines()
                    if l and not l.strip().startswith("#")
                ]
                if lines:
                    last_end = int(lines[-1].strip().split()[1])
                for _ in range(last_end):
                    try:
                        next(self.reader)
                        progress_bar.update(1)
                    except StopIteration:
                        break

        # 初始化计数器和状态变量
        scene_count = (
            len(glob.glob(os.path.join(output_video_dir, "*.mp4"))) if save_mp4 else 0
        )
        black_bar_skipped_count = 0
        prev_frame_pos = last_end
        video_read_out = False
        prev_segment, prev_segment_resized = None, None

        # 主处理循环
        while not video_read_out:
            try:
                # 传入 save_mp4 来决定是否在内存中保留高分辨率张量
                next_segment, next_segment_resized = self.get_video_frames(
                    segment_frames, keep_high_res=save_mp4
                )
            except ValueError:
                break

            video_read_out = next_segment_resized.shape[1] < segment_frames

            # 拼接上一段的剩余帧
            if prev_segment_resized is not None:
                if save_mp4:
                    next_segment = torch.cat([prev_segment, next_segment], dim=1)
                next_segment_resized = torch.cat(
                    [prev_segment_resized, next_segment_resized], dim=1
                )

            # 执行镜头检测
            scenes = self.segment_shot_detect(next_segment_resized, threshold=threshold)

            # 如果只检测到一个场景且视频未读完，继续累积
            if len(scenes) == 1 and not video_read_out:
                if save_mp4:
                    prev_segment = next_segment
                prev_segment_resized = next_segment_resized
                continue

            # 处理检测到的场景
            for scene in scenes[:-1] if len(scenes) > 1 else scenes:
                abs_start, abs_end = (
                    prev_frame_pos + scene[0],
                    prev_frame_pos + scene[1],
                )

                # 跳过过短的镜头
                if abs_end - abs_start < min_frames:
                    continue

                # ==========================
                # 无论是否保存视频，都记录镜头边界
                # ==========================
                with open(output_file, "a") as f:
                    f.write(f"{abs_start} {abs_end}\n")

                scene_count += 1

                # ==========================
                # 如果开启了保存视频
                # ==========================
                if save_mp4:
                    shot_frames = next_segment[0, scene[0] : scene[1]].cpu()

                    # 可选：裁剪到目标尺寸
                    if target_width is not None and target_height is not None:
                        cropped_frames = self._crop_frames_to_target_size(
                            shot_frames, target_width, target_height
                        )
                        if cropped_frames is None:
                            black_bar_skipped_count += 1
                            continue
                        shot_frames = cropped_frames

                    # 保存镜头视频
                    output_filename = os.path.join(
                        output_video_dir,
                        f"scene_{scene_count:03d}_{abs_start:06d}_{abs_end:06d}.mp4",
                    )
                    self._save_video(
                        shot_frames, output_filename, abs_start, abs_end, video_fps
                    )

            # 更新位置和缓存
            if scenes.size > 0:
                prev_frame_pos += scenes[-1][0]
                if save_mp4:
                    prev_segment = next_segment[:, scenes[-1][0] :]
                prev_segment_resized = next_segment_resized[:, scenes[-1][0] :]
                progress_bar.update(
                    scenes[-1][0] if not video_read_out else scenes[-1][1]
                )

        progress_bar.close()

        # 全部检测完成，写入结束符，便于下次直接跳过
        with open(output_file, "a") as f:
            f.write("# COMPLETE\n")

        # 打印处理结果
        print(f"\n镜头检测完成！")
        print(f"总计检测到: {scene_count} 个镜头。结果已存入 {output_file}")
        if save_mp4 and black_bar_skipped_count > 0:
            print(f"因黑边跳过保存: {black_bar_skipped_count} 个镜头")

        return scene_count


class TransNetV2(nn.Module):
    def __init__(self,
                 F=16, L=3, S=2, D=1024,
                 use_many_hot_targets=True,
                 use_frame_similarity=True,
                 use_color_histograms=True,
                 use_mean_pooling=False,
                 dropout_rate=0.5,
                 use_convex_comb_reg=False,  
                 use_resnet_features=False,  
                 use_resnet_like_top=False,  
                 frame_similarity_on_last_layer=False):  
        super(TransNetV2, self).__init__()

        if use_resnet_features or use_resnet_like_top or use_convex_comb_reg or frame_similarity_on_last_layer:
            raise NotImplementedError("Some options not implemented in Pytorch version of Transnet!")

        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.)] +
            [StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2 ** i) for i in range(1, L)]
        )

        self.frame_sim_layer = FrameSimilarity(
            sum([(F * 2 ** i) * 4 for i in range(L)]), lookup_window=101, output_dim=128, similarity_dim=128, use_bias=True
        ) if use_frame_similarity else None
        self.color_hist_layer = ColorHistograms(
            lookup_window=101, output_dim=128
        ) if use_color_histograms else None

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  # 3x6 for spatial dimensions
        if use_frame_similarity: output_dim += 128
        if use_color_histograms: output_dim += 128

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1) if use_many_hot_targets else None

        self.use_mean_pooling = use_mean_pooling
        self.eval()

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor) and list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == torch.uint8, \
            "incorrect input type and/or shape"
            
        # [OPTIMIZED] 增加 .contiguous() 确保内存连续性，加速后续访存
        x = inputs.permute([0, 4, 1, 2, 3]).contiguous().float()
        x = x.div_(255.)

        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        if self.frame_sim_layer is not None:
            x = torch.cat([self.frame_sim_layer(block_features), x], 2)

        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], 2)

        x = self.fc1(x)
        x = functional.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        one_hot = self.cls_layer1(x)

        if self.cls_layer2 is not None:
            return one_hot, {"many_hot": self.cls_layer2(x)}

        return one_hot

    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
        # [FIXED] 将 uint8 改为 int8，防止 np.diff 计算 0 - 1 时发生下溢出变成 255
        predictions = (predictions > threshold).astype(np.int8)

        if len(predictions) == 0:
            return np.array([], dtype=np.int32)

        diff = np.diff(predictions)
        
        # 寻找状态改变的索引
        starts = np.where(diff == -1)[0] + 1
        ends = np.where(diff == 1)[0] + 1
        
        # 处理边界情况
        if predictions[0] == 0:
            starts = np.insert(starts, 0, 0)
        if predictions[-1] == 0:
            ends = np.append(ends, len(predictions) - 1)
            
        # 兼容原始逻辑：如果全是 1 (没有场景)
        if len(starts) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)
            
        scenes = np.column_stack((starts, ends))
        return scenes.astype(np.int32)

    @staticmethod
    def visualize_predictions(frames: np.ndarray, predictions):
        from PIL import Image, ImageDraw

        if isinstance(predictions, np.ndarray):
            predictions = [predictions]

        ih, iw, ic = frames.shape[1:]
        width = 25

        pad_with = width - len(frames) % width if len(frames) % width != 0 else 0
        frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)), (0, 0)])

        predictions = [np.pad(x, (0, pad_with)) for x in predictions]
        height = len(frames) // width

        img = frames.reshape([height, width, ih + 1, iw + len(predictions), ic])
        img = np.concatenate(np.split(
            np.concatenate(np.split(img, height), axis=2)[0], width
        ), axis=2)[0, :-1]

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        for i, pred in enumerate(zip(*predictions)):
            x, y = i % width, i // width
            x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

            for j, p in enumerate(pred):
                color = [0, 0, 0]
                color[(j + 1) % 3] = 255

                value = np.round(p * (ih - 1))
                if value != 0:
                    draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=1)
        return img


class StackedDDCNNV2(nn.Module):
    def __init__(self,
                 in_filters,
                 n_blocks,
                 filters,
                 shortcut=True,
                 use_octave_conv=False,  
                 pool_type="avg",
                 stochastic_depth_drop_prob=0.0):
        super(StackedDDCNNV2, self).__init__()

        if use_octave_conv:
            raise NotImplementedError("Octave convolution not implemented in Pytorch version of Transnet!")

        assert pool_type == "max" or pool_type == "avg"
        if use_octave_conv and pool_type == "max":
            print("WARN: Octave convolution was designed with average pooling, not max pooling.")

        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList([
            DilatedDCNNV2(in_filters if i == 1 else filters * 4, filters, octave_conv=use_octave_conv,
                          activation=functional.relu if i != n_blocks else None) for i in range(1, n_blocks + 1)
        ])
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool_type == "max" else nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs):
        x = inputs
        shortcut = None

        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x

        x = functional.relu(x)

        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.:
                if self.training:
                    if random.random() < self.stochastic_depth_drop_prob:
                        x = shortcut
                    else:
                        x = x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x += shortcut

        x = self.pool(x)
        return x


class DilatedDCNNV2(nn.Module):
    def __init__(self,
                 in_filters,
                 filters,
                 batch_norm=True,
                 activation=None,
                 octave_conv=False):  
        super(DilatedDCNNV2, self).__init__()

        if octave_conv:
            raise NotImplementedError("Octave convolution not implemented in Pytorch version of Transnet!")

        assert not (octave_conv and batch_norm)

        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs):
        conv1 = self.Conv3D_1(inputs)
        conv2 = self.Conv3D_2(inputs)
        conv3 = self.Conv3D_4(inputs)
        conv4 = self.Conv3D_8(inputs)

        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class Conv3DConfigurable(nn.Module):
    def __init__(self,
                 in_filters,
                 filters,
                 dilation_rate,
                 separable=True,
                 octave=False,  
                 use_bias=True,
                 kernel_initializer=None):  
        super(Conv3DConfigurable, self).__init__()

        if octave:
            raise NotImplementedError("Octave convolution not implemented in Pytorch version of Transnet!")
        if kernel_initializer is not None:
            raise NotImplementedError("Kernel initializers are not implemented in Pytorch version of Transnet!")

        assert not (separable and octave)

        if separable:
            conv1 = nn.Conv3d(in_filters, 2 * filters, kernel_size=(1, 3, 3),
                              dilation=(1, 1, 1), padding=(0, 1, 1), bias=False)
            conv2 = nn.Conv3d(2 * filters, filters, kernel_size=(3, 1, 1),
                              dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 0, 0), bias=use_bias)
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(in_filters, filters, kernel_size=3,
                             dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 1, 1), bias=use_bias)
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class FrameSimilarity(nn.Module):
    def __init__(self,
                 in_filters,
                 similarity_dim=128,
                 lookup_window=101,
                 output_dim=128,
                 stop_gradient=False,  
                 use_bias=False):
        super(FrameSimilarity, self).__init__()

        if stop_gradient:
            raise NotImplementedError("Stop gradient not implemented in Pytorch version of Transnet!")

        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)

        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def forward(self, inputs):
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        x = torch.transpose(x, 1, 2)

        x = self.projection(x)
        x = functional.normalize(x, p=2, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  
        
        # [OPTIMIZED] 使用 as_strided 提取对角线滑动窗口，取代高昂开销的高级索引
        pad_size = (self.lookup_window - 1) // 2
        similarities_padded = functional.pad(similarities, [pad_size, pad_size])
        similarities_padded = similarities_padded.contiguous()
        
        stride_b, stride_t, stride_l = similarities_padded.stride()
        similarities = torch.as_strided(
            similarities_padded,
            size=(batch_size, time_window, self.lookup_window),
            stride=(stride_b, stride_t + stride_l, stride_l)
        )
        
        return functional.relu(self.fc(similarities))


class ColorHistograms(nn.Module):
    def __init__(self,
                 lookup_window=101,
                 output_dim=None):
        super(ColorHistograms, self).__init__()

        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    @staticmethod
    def compute_color_histograms(frames):
        frames = frames.int()

        def get_bin(frames):
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = R >> 5, G >> 5, B >> 5
            return (R << 6) + (G << 3) + B

        batch_size, time_window, height, width, no_channels = frames.shape
        assert no_channels == 3
        frames_flatten = frames.view(batch_size * time_window, height * width, 3)

        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = (torch.arange(0, batch_size * time_window, device=frames.device) << 9).view(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).view(-1)

        histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        histograms.scatter_add_(0, binned_values, torch.ones(len(binned_values), dtype=torch.int32, device=frames.device))

        histograms = histograms.view(batch_size, time_window, 512).float()
        histograms_normalized = functional.normalize(histograms, p=2, dim=2)
        return histograms_normalized

    def forward(self, inputs):
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  
        
        # [OPTIMIZED] 同样应用 as_strided 优化技巧
        pad_size = (self.lookup_window - 1) // 2
        similarities_padded = functional.pad(similarities, [pad_size, pad_size])
        similarities_padded = similarities_padded.contiguous()
        
        stride_b, stride_t, stride_l = similarities_padded.stride()
        similarities = torch.as_strided(
            similarities_padded,
            size=(batch_size, time_window, self.lookup_window),
            stride=(stride_b, stride_t + stride_l, stride_l)
        )

        if self.fc is not None:
            return functional.relu(self.fc(similarities))
        return similarities


if __name__ == "__main__":
    import argparse
    import glob

    # 设置命令行参数
    parser = argparse.ArgumentParser(description="批量检测视频镜头并自动切分为 MP4 片段")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="Datasets/Movies", 
        help="包含电影文件的输入目录 (默认: Datasets/Movies)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="Datasets/AVAGen", 
        help="保存切分后镜头的根目录 (默认: Datasets/AVAGen)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None, 
        help="计算设备 (例如 'cuda:0', 默认自动检测)"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.2, 
        help="镜头检测阈值，越小越敏感 (默认: 0.2)"
    )
    parser.add_argument(
        "--min_frames", 
        type=int, 
        default=24, 
        help="保留的最短镜头帧数，过滤掉过短的闪烁镜头 (默认: 15)"
    )
    args = parser.parse_args()

    # 1. 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"[错误] 输入目录不存在: {args.input_dir}")
        exit(1)

    # 2. 获取所有的 .mp4 文件 (Pipeline 中的 read_video 限制了必须为 mp4)
    video_files = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")))
    if not video_files:
        print(f"[提示] 在 {args.input_dir} 中没有找到 .mp4 文件。")
        exit(0)

    print(f"[INFO] 找到 {len(video_files)} 个视频文件，准备初始化模型...")

    # 3. 初始化 Pipeline (放在循环外，只需加载一次权重)
    pipeline = TransNetV2Pipeline(device=args.device)

    # 4. 遍历处理每个视频
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        # 为每部电影创建一个专属的子目录
        movie_output_dir = os.path.join(args.output_dir, video_name)
        
        print("-" * 50)
        print(f"[处理中] 视频名称: {video_name}")
        print(f"[处理中] 视频路径: {video_path}")
        print(f"[处理中] 输出路径: {movie_output_dir}")

        try:
            # 读取视频
            pipeline.read_video(video_path)
            
            # 执行镜头检测与切分
            shot_count = pipeline.shot_detect(
                output_dir=movie_output_dir,
                threshold=args.threshold,
                min_frames=args.min_frames,
                save_mp4=True
            )
            print(f"[成功] {video_name} 处理完毕，共提取 {shot_count} 个镜头。")
            
        except Exception as e:
            print(f"[失败] 处理 {video_name} 时发生错误: {e}")
            
    print("=" * 50)
    print("[INFO] 所有视频处理任务已完成！")