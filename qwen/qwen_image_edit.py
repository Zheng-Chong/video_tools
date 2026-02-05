from diffusers import QwenImageEditPlusPipeline, AutoModel, BitsAndBytesConfig
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import warnings
import logging

# ================= 配置区域 =================
# 输入目录：qwen3_vl.py 裁切出来的人物图像所在目录
INPUT_ROOT = Path("Datasets/AVAGen-480P-Character-bbox")
# 模型路径
MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
# 输出分辨率
OUTPUT_WIDTH = 768
OUTPUT_HEIGHT = 1024
# Prompt
PROMPT = "转化为该人物的标准站姿的全身照片，白色背景，3:4 长宽比，注意保持服装细节一致性，人物正背和原图保持一致"

# ================= 屏蔽警告 =================
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

# ================= 初始化 Pipeline =================
def init_pipeline():
    """初始化 QwenImageEditPlusPipeline"""
    print(f"Loading model from {MODEL_ID}...")
    
    quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    transformer = AutoModel.from_pretrained(
        MODEL_ID, 
        subfolder="transformer", 
        quantization_config=quant_config, 
        torch_dtype=torch.bfloat16
    )
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID, 
        transformer=transformer, 
        torch_dtype=torch.bfloat16, 
        device_map="balanced"
    )
    
    print("Model loaded successfully.")
    return pipeline

# ================= 处理单张图像 =================
def process_single_image(pipeline, input_image_path, output_image_path):
    """
    处理单张人物图像，转换为标准站姿参考图
    
    Args:
        pipeline: QwenImageEditPlusPipeline 实例
        input_image_path: 输入图像路径
        output_image_path: 输出图像路径
    Returns:
        成功返回True，失败返回False
    """
    try:
        # 确保输出目录存在
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果输出文件已存在，跳过
        if output_image_path.exists():
            return True
        
        # 加载输入图像
        input_image = Image.open(input_image_path).convert("RGB")
        
        # 使用 pipeline 进行图像编辑
        result = pipeline(
            prompt=PROMPT,
            image=input_image,
            height=OUTPUT_HEIGHT,
            width=OUTPUT_WIDTH,
            num_inference_steps=50,  # 可以根据需要调整
        )
        
        # 获取生成的图像
        output_image = result.images[0]
        
        # 保存图像
        output_image.save(output_image_path, quality=95)
        
        return True
    except Exception as e:
        print(f"Error processing {input_image_path}: {e}")
        return False

# ================= 主函数 =================
def main():
    """主处理函数"""
    # 初始化 pipeline
    pipeline = init_pipeline()
    
    # 查找所有裁切出来的人物图像（格式：*_person*.jpg）
    print(f"Scanning for person images in {INPUT_ROOT}...")
    person_images = list(INPUT_ROOT.rglob("*_person*.jpg"))
    
    # 过滤掉已经处理过的文件
    todo_images = []
    for img_path in person_images:
        # 计算输出路径：在原文件名后添加 _norm 后缀
        # 例如：clip_000054_person1.jpg -> clip_000054_person1_norm.jpg
        output_path = img_path.parent / f"{img_path.stem}_norm{img_path.suffix}"
        
        # 如果输出文件不存在，加入待处理列表
        if not output_path.exists():
            todo_images.append((img_path, output_path))
    
    print(f"Found {len(todo_images)} images to process (skipping {len(person_images) - len(todo_images)} already processed).")
    
    if not todo_images:
        print("All images already processed!")
        return
    
    # 处理所有图像
    success_count = 0
    error_count = 0
    
    for input_path, output_path in tqdm(todo_images, desc="Processing images"):
        if process_single_image(pipeline, input_path, output_path):
            success_count += 1
        else:
            error_count += 1
    
    print(f"\nProcessing completed!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output images saved in the same directories as input images with '_norm' suffix.")

if __name__ == "__main__":
    main()
