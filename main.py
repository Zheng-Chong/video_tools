"""
TransNetV2 视频镜头检测使用示例
"""
from transnet import TransNetV2Pipeline

# 视频文件路径
mp4_path = "/home/chongzheng_p23/projects/Movie-Tools/Datasets/MovieChat/Mission.Impossible.-.The.Final.Reckoning.2025/Mission.Impossible.-.The.Final.Reckoning.2025.720p.WEBRip.x264.AAC-[YTS.MX].mp4"

# 输出目录
output_dir = "./output/mission_impossible"


def main():
    # 1. 初始化 TransNetV2 管道
    #    - 自动从 HuggingFace 下载模型（首次运行）
    #    - 自动选择 GPU/CPU
    print("正在初始化 TransNetV2 模型...")
    pipeline = TransNetV2Pipeline(
        model_path="/home/chongzheng_p23/projects/Movie-Tools/Models/TransNetV2",
        device="cuda")
    
    # 2. 读取视频文件
    print(f"正在加载视频: {mp4_path}")
    pipeline.read_video(mp4_path)
    
    # 3. 执行镜头检测并分割保存
    print("开始镜头检测...")
    shot_count = pipeline.shot_detect(
        output_dir=output_dir,
        threshold=0.2,       # 检测阈值，越小越敏感
        min_frames=120,      # 最小镜头帧数（约 5 秒 @24fps）
        segment_frames=1000  # 每批处理帧数
    )
    
    print(f"\n处理完成！共检测到 {shot_count} 个镜头")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
