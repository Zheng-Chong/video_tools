import easyocr

from tools.video_io import iter_sampled_frames

class EasyOCRCreditDetectorPipeline:
    def __init__(self, 
                 sample_interval_sec=1.0, 
                 min_text_boxes=1, 
                 min_box_area=200, 
                 dense_frame_ratio=0.3):
        """
        初始化 EasyOCR 检测管道
        """
        # 初始化 EasyOCR Reader
        # 默认加载简体中文和英文。verbose=False 用于屏蔽初始化时烦人的普通日志
        # gpu=True 默认开启 GPU 加速，如果没有 GPU 它会自动回退到 CPU
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=True, verbose=False)
        
        self.sample_interval_sec = sample_interval_sec
        self.min_text_boxes = min_text_boxes
        self.min_box_area = min_box_area
        self.dense_frame_ratio = dense_frame_ratio

    def _is_valid_box(self, box):
        """
        计算文本框面积，过滤噪点。
        注意: EasyOCR 的 detect 方法返回的 box 格式是 [x_min, x_max, y_min, y_max]
        这比 PaddleOCR 的多边形坐标计算起来更简单。
        """
        try:
            x_min, x_max, y_min, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            return (width * height) > self.min_box_area
        except Exception:
            return False

    def process_clip(self, video_path):
        """
        处理传入的视频片段，返回判定结果和详细统计信息
        """
        total_sampled_frames = 0
        dense_text_frames = 0

        for total_sampled_frames, frame in iter_sampled_frames(
                video_path,
                sample_interval_sec=self.sample_interval_sec,
                default_fps=24.0,
            ):
                # 核心优化：使用 detect() 只做文本检测，跳过识别步骤。
                # detect() 返回格式为一个元组: (horizontal_list, free_list)
                result = self.reader.detect(frame)

                valid_box_count = 0
                # 解析 EasyOCR 的输出格式
                # result[0] 是水平文本框列表。对于单张图片，它的结构通常是 [[box1, box2, ...]]
                if result and len(result[0]) > 0:
                    boxes = result[0][0]
                    for box in boxes:
                        if self._is_valid_box(box):
                            valid_box_count += 1

                # 若检测到有效文本框（有字），记为一次“有文字帧”
                if valid_box_count >= self.min_text_boxes:
                    dense_text_frames += 1
                    # 早停：已判定为片头/片尾，无需继续检测
                    report = {
                        "total_sampled": total_sampled_frames,
                        "dense_frames": dense_text_frames,
                        "actual_ratio": 1.0,
                        "threshold": ">=1 frame with text",
                        "early_stopped": True,
                    }
                    return True, report

        if total_sampled_frames == 0:
            return False, {"error": "视频过短或读取失败"}

        # 宽松判定：只要有一帧检测到文字就判断为片头/片尾
        actual_ratio = dense_text_frames / total_sampled_frames
        is_credit = dense_text_frames >= 1

        report = {
            "total_sampled": total_sampled_frames,
            "dense_frames": dense_text_frames,
            "actual_ratio": round(actual_ratio, 3),
            "threshold": ">=1 frame with text",
            "early_stopped": False
        }

        return is_credit, report

if __name__ == "__main__":
    # 1. 实例化新版 EasyOCR Pipeline
    pipeline = EasyOCRCreditDetectorPipeline(
        sample_interval_sec=1.0, 
        min_text_boxes=1,    # 画面中至少有1个有效文字块即视为“有字”
        dense_frame_ratio=0.4 # 已弃用，现采用宽松判定：任一帧有字即判为片头/尾
    )

    # 2. 传入视频片段进行测试
    test_video = "Datasets/AVAGen/Bad.Man.2025.720p.WEBRip.x264.AAC-[YTS.MX]/shots/scene_002_001574_001773.mp4"
    
    # 替换为实际用于测试的路径，避免直接运行报错
    import os
    if os.path.exists(test_video):
        print(f"开始分析视频: {test_video} ...")
        is_opening_or_ending, details = pipeline.process_clip(test_video)
        
        # 3. 输出结果
        if is_opening_or_ending:
            print("✅ 判定结果: 该片段【是】片头或片尾")
        else:
            print("❌ 判定结果: 该片段【不是】片头或片尾 (可能是正片内容)")
            
        print(f"详细数据: {details}")
    else:
        print(f"⚠️ 测试视频文件不存在，请检查路径: {test_video}")