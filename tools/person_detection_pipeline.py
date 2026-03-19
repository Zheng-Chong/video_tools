"""
人物检测 Pipeline
================

检测视频片段中是否包含人物。基于 torchvision 的 Faster R-CNN（COCO 预训练）
进行人体检测，采样帧并判断是否有人物出现。

与 ocr_pipeline 保持一致的接口：process_clip(video_path) -> (has_person, report)
"""
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights

from tools.video_io import iter_sampled_frames

# COCO 数据集中 person 的类别 ID
COCO_PERSON_CLASS_ID = 1


class PersonDetectionPipeline:
    def __init__(
        self,
        sample_interval_sec: float = 1.0,
        min_person_frames: int = 1,
        confidence_threshold: float = 0.5,
        device: str | None = None,
    ):
        """
        初始化人物检测 Pipeline。

        Args:
            sample_interval_sec: 采样间隔（秒），每隔多少秒取一帧
            min_person_frames: 至少多少帧检测到人物才判定为「有人」
            confidence_threshold: 检测置信度阈值，低于此值的框忽略
            device: 推理设备，如 "cuda" / "cpu"，None 时自动选择
        """
        self.sample_interval_sec = sample_interval_sec
        self.min_person_frames = min_person_frames
        self.confidence_threshold = confidence_threshold

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        self.transforms = weights.transforms()

    def _detect_person_in_frame(self, frame) -> bool:
        """
        对单帧进行人物检测，返回是否检测到人物。
        frame: BGR 格式的 numpy 数组 (H, W, 3)
        """
        # BGR -> RGB -> PIL Image（transforms 需要 PIL 输入）
        rgb = frame[:, :, ::-1]
        pil_img = Image.fromarray(rgb)
        # 转为 tensor [C, H, W]，并做模型要求的预处理
        img_tensor = self.transforms(pil_img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)

        if not outputs or len(outputs) == 0:
            return False

        boxes = outputs[0]["boxes"]
        labels = outputs[0]["labels"]
        scores = outputs[0]["scores"]

        for i, label in enumerate(labels):
            if label.item() == COCO_PERSON_CLASS_ID and scores[i].item() >= self.confidence_threshold:
                return True
        return False

    def process_clip(self, video_path: str) -> tuple[bool, dict]:
        """
        处理传入的视频片段，返回是否包含人物及详细统计。

        Returns:
            (has_person, report): 是否有人物，以及检测报告
        """
        total_sampled_frames = 0
        person_frames = 0

        for total_sampled_frames, frame in iter_sampled_frames(
                video_path,
                sample_interval_sec=self.sample_interval_sec,
                default_fps=24.0,
            ):
                if self._detect_person_in_frame(frame):
                    person_frames += 1
                    # 早停：已满足「有人」判定
                    if person_frames >= self.min_person_frames:
                        report = {
                            "total_sampled": total_sampled_frames,
                            "person_frames": person_frames,
                            "actual_ratio": 1.0 if total_sampled_frames > 0 else 0.0,
                            "threshold": f">={self.min_person_frames} frame(s) with person",
                            "early_stopped": True,
                        }
                        return True, report

        if total_sampled_frames == 0:
            return False, {"error": "视频过短或读取失败"}

        actual_ratio = person_frames / total_sampled_frames
        has_person = person_frames >= self.min_person_frames

        report = {
            "total_sampled": total_sampled_frames,
            "person_frames": person_frames,
            "actual_ratio": round(actual_ratio, 3),
            "threshold": f">={self.min_person_frames} frame(s) with person",
            "early_stopped": False,
        }

        return has_person, report


if __name__ == "__main__":
    import os

    pipeline = PersonDetectionPipeline(
        sample_interval_sec=1.0,
        min_person_frames=1,
        confidence_threshold=0.5,
    )

    test_video = "Datasets/AVAGen/Bad.Man.2025.720p.WEBRip.x264.AAC-[YTS.MX]/shots/scene_002_001574_001773.mp4"

    if os.path.exists(test_video):
        print(f"开始分析视频: {test_video} ...")
        has_person, report = pipeline.process_clip(test_video)

        if has_person:
            print("✅ 判定结果: 该片段【包含】人物")
        else:
            print("❌ 判定结果: 该片段【不包含】人物")

        print(f"详细数据: {report}")
    else:
        print(f"⚠️ 测试视频文件不存在，请检查路径: {test_video}")
