import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import warnings
import logging
import torch
import json

# 保持原有的 import 逻辑
from diffusers import QwenImageEditPlusPipeline

# ================= 配置区域 =================
INPUT_ROOT = Path("Datasets/AVAGen-480P-Character-bbox")
MODEL_ID = "Models/Qwen-Image-Edit-2511-4bit"
OUTPUT_WIDTH = 896
OUTPUT_HEIGHT = 1152
NEGATIVE_PROMPT = "  "

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

def init_pipeline(rank):
    """初始化 Pipeline。注意：配合 CUDA_VISIBLE_DEVICES 使用时，设备始终为 cuda:0"""
    print(f"[Rank {rank}] Loading model...")
    
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16
    )
    # pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    pipeline.vae.enable_tiling()
    
    # 核心修改：因为 Shell 脚本里指定了 CUDA_VISIBLE_DEVICES
    # 对每个进程来说，它能看到的唯一一张卡索引都是 0
    device = "cuda:0" 
    pipeline.to(device)
    
    # 针对 OOM 的额外优化：清理显存缓存
    torch.cuda.empty_cache()
    
    return pipeline, device

def generate_prompt(face_visible, shot_type):
    """根据 face_visible 和 shot_type 生成动态 prompt"""
    # 根据 face_visible 决定朝向
    orientation = "正面" if face_visible else "背面"
    
    # 根据 shot_type 决定拍摄类型
    shot_type_map = {
        "half_body": "半身",
        "full_body": "全身",
        "close_up": "特写",
        "upper_body": "上半身",
        "medium_shot": "中景",
        "long_shot": "远景"
    }
    shot_description = shot_type_map.get(shot_type, "全身")  # 默认为全身
    
    prompt = f"基于原图人物进行编辑。转化为标准的{shot_description}直立站姿照片。最重要约束：生成的人物必须是{orientation}视角。绝不可旋转人物身体。保持所有服装、头发和配件细节与原图完美一致。纯白色背景。3:4纵向比例。"
    
    return prompt

def load_orientation_info(image_path):
    """从对应的 JSON 文件中加载人物朝向信息"""
    # 从图片路径推断 JSON 文件路径
    # 例如: 2.Fast.2.Furious.2003/clip_000019_person1.jpg -> 2.Fast.2.Furious.2003/clip_000019.json
    image_name = image_path.name
    # 提取 clip 名称（去掉 _personX.jpg）
    if "_person" in image_name:
        clip_name = image_name.split("_person")[0]
        json_name = f"{clip_name}.json"
        json_path = image_path.parent / json_name
        
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    character_orientations = data.get("character_orientations", {})
                    # 使用相对路径作为 key
                    relative_path = str(image_path.relative_to(INPUT_ROOT))
                    orientation_info = character_orientations.get(relative_path)
                    if orientation_info:
                        return orientation_info.get("face_visible", True), orientation_info.get("shot_type", "full_body")
            except Exception as e:
                print(f"Warning: Failed to load orientation info from {json_path}: {e}")
    
    # 默认值：正面、全身
    return True, "full_body"

def process_single_image(pipeline, device, input_image_path, output_image_path, prompt):
    try:
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        if output_image_path.exists():
            return True
        
        input_image = Image.open(input_image_path).convert("RGB")
        
        with torch.inference_mode():
            # 确保输入数据在对应设备上
            print(f"Prompt: {prompt}")
            result = pipeline(
                prompt=prompt,
                image=input_image,
                height=OUTPUT_HEIGHT,
                width=OUTPUT_WIDTH,
                true_cfg_scale=4.0,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=25, 
            )
        
        output_image = result.images[0]
        output_image.save(output_image_path, quality=95)
        return True
    except Exception as e:
        print(f"\nError on Rank {device} processing {input_image_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0, help="当前进程的索引 (卡号)")
    parser.add_argument("--world_size", type=int, default=1, help="总进程数 (总卡数)")
    args = parser.parse_args()

    # 1. 扫描所有文件
    person_images = sorted(list(INPUT_ROOT.rglob("*_person*.jpg"))) # 排序确保分片逻辑一致
    # 过滤掉尾缀是 norm 的文件
    person_images = [img for img in person_images if "_norm" not in img.stem]
    
    # 2. 过滤已存在的文件，并加载朝向信息
    todo_tasks = []
    for img_path in person_images:
        output_path = img_path.parent / f"{img_path.stem}_norm{img_path.suffix}"
        if not output_path.exists():
            # 加载朝向信息并生成 prompt
            face_visible, shot_type = load_orientation_info(img_path)
            prompt = generate_prompt(face_visible, shot_type)
            todo_tasks.append((img_path, output_path, prompt))

    # 3. 分片逻辑：每个进程只处理自己那部分
    # 按照 rank 取模分配
    my_tasks = todo_tasks[args.rank::args.world_size]
    
    print(f"Rank {args.rank}/{args.world_size}: Processing {len(my_tasks)} images.")

    if not my_tasks:
        print(f"Rank {args.rank}: No tasks to perform.")
        return

    # 4. 初始化 Pipeline
    pipeline, device = init_pipeline(args.rank)
    
    # 5. 执行处理
    success_count = 0
    for input_path, output_path, prompt in tqdm(my_tasks, desc=f"Rank {args.rank}"):
        if process_single_image(pipeline, device, input_path, output_path, prompt):
            success_count += 1
    
    print(f"Rank {args.rank} finished. Success: {success_count}/{len(my_tasks)}")

if __name__ == "__main__":
    main()