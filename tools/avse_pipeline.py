import os
import shutil
import tempfile
import subprocess
import numpy as np
import cv2
import mediapipe as mp  # 只导入最顶层
from clearvoice import ClearVoice


def build_avse_model():
    """构建并返回 AVSE 模型实例（可在外部复用，避免重复加载）。"""
    print("🧠 正在加载 ClearerVoice 多模态模型进行定向音频提取...")
    return ClearVoice(
        task="target_speaker_extraction",
        model_names=["AV_MossFormer2_TSE_16K"],
    )


def crop_target_face_video(input_mp4, output_face_mp4, target_size=(160, 160)):
    """
    第一步：从原始视频中检测并裁剪目标人物的面部/唇部
    """
    print(f"🎬 开始处理视觉画面，裁剪目标人脸...")
    cap = cv2.VideoCapture(input_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化 MediaPipe 人脸检测（使用 with 确保资源正确释放）
    mp_face_detection = mp.solutions.face_detection

    # 设置输出视频流
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_face_mp4, fourcc, fps, target_size)

    last_bbox = None

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 将 BGR 转为 RGB 以适配 MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)

            if results.detections:
                # 策略：如果有多个人脸，选择检测框面积最大的那个人脸（主讲人）
                largest_face = max(
                    results.detections,
                    key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height
                )
                bbox = largest_face.location_data.relative_bounding_box

                # 转换为绝对坐标
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # 以人脸中心为基准，向外扩展一定比例（包含完整的下巴和唇部）
                cx, cy = x + w // 2, y + h // 2
                side_length = int(max(w, h) * 1.5)  # 1.5倍的包围盒
                half_side = side_length // 2

                x1, y1 = max(0, cx - half_side), max(0, cy - half_side)
                x2, y2 = min(width, cx + half_side), min(height, cy + half_side)

                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size > 0:
                    # 统一缩放至模型所需的固定尺寸
                    face_crop_resized = cv2.resize(face_crop, target_size)
                    out_video.write(face_crop_resized)
                    last_bbox = (x1, y1, x2, y2)
            else:
                # 如果某几帧因为低头等原因没检测到，使用上一帧的位置（保持时序连贯）
                if last_bbox:
                    x1, y1, x2, y2 = last_bbox
                    # 确保 bbox 有效（避免 x2<=x1 或 y2<=y1 导致空裁剪）
                    if x2 > x1 and y2 > y1:
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            face_crop_resized = cv2.resize(face_crop, target_size)
                            out_video.write(face_crop_resized)
                        else:
                            out_video.write(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
                    else:
                        out_video.write(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
                else:
                    # 如果一开始就没检测到，写入黑帧
                    out_video.write(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))

    cap.release()
    out_video.release()
    print(f"✅ 人脸视频提取完成: {output_face_mp4}")


def run_avse_clip(
    input_mp4_path,
    output_clean_wav_path,
    avse_model=None,
):
    """
    处理单个 clip 的 AVSE 提取流程。
    - 输入：单个 clip 视频（mp4）
    - 输出：对应的干净人声（wav）
    - 可选：传入已初始化的 avse_model 以复用模型、减少重复加载开销
    """
    model = avse_model if avse_model is not None else build_avse_model()

    # 使用唯一临时目录，避免并行运行时文件冲突
    temp_dir = tempfile.mkdtemp(prefix="avse_pipeline_")
    temp_mixed_wav = os.path.join(temp_dir, "mixed_audio.wav")
    temp_face_mp4 = os.path.join(temp_dir, "face_only.mp4")

    try:
        # 1. 提取 16kHz 的混合音频
        print(f"🎵 正在从 {input_mp4_path} 提取底层混合音频...")
        result = subprocess.run(
            [
                "ffmpeg", "-i", input_mp4_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                temp_mixed_wav, "-y"
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 提取音频失败 (exit={result.returncode}): {result.stderr}")

        # 2. 提取视觉条件（仅人脸画面，生成的是无声视频）
        crop_target_face_video(input_mp4_path, temp_face_mp4)

        # [新增步骤] 2.5 将无声人脸视频与 16kHz 混合音频封装对齐
        temp_av_avi = os.path.join(temp_dir, "av_mixed.avi")
        print("🔀 正在将人脸视频与音频封装为多模态输入文件...")
        mux_result = subprocess.run(
            [
                "ffmpeg", "-i", temp_face_mp4, "-i", temp_mixed_wav,
                "-c:v", "copy", "-c:a", "pcm_s16le", temp_av_avi, "-y"
            ],
            capture_output=True,
            text=True,
        )
        if mux_result.returncode != 0:
            raise RuntimeError(f"ffmpeg 音视频封装失败: {mux_result.stderr}")

        # 3. 运行 ClearerVoice 多模态提取（AV_MossFormer2_TSE_16K 强制要求 online_write=True）
        model(
            input_path=temp_av_avi,
            online_write=True,                 # 满足底层源码的强制要求
            output_path=output_clean_wav_path  # 让 ClearVoice 内部直接把音频保存到此路径
        )
        print(f"🎉 完美！目标角色的干净人声已保存至: {output_clean_wav_path}")

    finally:
        # 5. 清理临时目录，保持工作区干净
        print("🧹 清理临时缓存文件...")
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_avse_pipeline(input_mp4_path, output_clean_wav_path):
    """
    兼容旧接口：单次处理一个输入视频并输出干净人声。
    """
    run_avse_clip(
        input_mp4_path=input_mp4_path,
        output_clean_wav_path=output_clean_wav_path,
        avse_model=None,
    )

# ---------------- 使用示例 ----------------
if __name__ == "__main__":
    # 你的多人物电影场景片段
    input_video = "Datasets/AVAGen-5s/Bad.Man.2025.720p.WEBRip.x264.AAC-[YTS.MX]/shots/scene_013_007407_007560.mp4"
    # 最终输出的干净人声
    final_clean_audio = "actor_clean_voice.wav"
    
    # 确保输入文件存在后再运行
    if os.path.exists(input_video):
        run_avse_pipeline(input_video, final_clean_audio)
    else:
        print(f"找不到输入文件 {input_video}，请检查路径。")