import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import warnings
import logging
import torch
import torch.multiprocessing as mp

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

def init_pipeline(device="cuda:0"):
    """初始化 Pipeline。支持传入完整 device 字符串或 GPU 序号。"""
    if isinstance(device, int):
        device = f"cuda:{device}"
    print(f"[QwenImageEdit] Loading model to {device}...")

    # 对当前 diffusers 版本，QwenImageEditPlusPipeline 的 device_map 不支持 "cuda:0" 这类具体卡号。
    # 统一先加载到 CPU，再显式迁移到目标设备，避免初始化阶段意外占用默认 GPU。
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    
    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.enable_tiling()
    
    pipeline.to(device)
    
    # 清理显存缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return pipeline, device


class QwenImageEditRunner:
    """供长生命周期服务复用的单实例包装器。"""

    def __init__(self, device="cuda:0", default_face_visible=True, default_shot_type=None):
        self.device = device
        self.default_face_visible = default_face_visible
        self.default_shot_type = default_shot_type
        self.pipeline = None

    def ensure_loaded(self):
        if self.pipeline is None:
            self.pipeline, self.device = init_pipeline(self.device)
        return self.pipeline, self.device

    def process(
        self,
        input_image_path,
        output_image_path=None,
        face_visible=None,
        shot_type=None,
        prompt=None,
    ):
        input_path = Path(input_image_path)
        if output_image_path is None:
            output_path = input_path.parent / f"{input_path.stem}_norm{input_path.suffix}"
        else:
            output_path = Path(output_image_path)

        if face_visible is None:
            face_visible = self.default_face_visible
        if shot_type is None:
            shot_type = self.default_shot_type
        if prompt is None:
            prompt = generate_prompt(face_visible, shot_type)

        pipeline, device = self.ensure_loaded()
        return process_single_image(pipeline, device, input_path, output_path, prompt)

def generate_prompt(face_visible, shot_type=None):
    """根据 face_visible 和 shot_type 生成动态 prompt。shot_type 为空时不指定景别。"""
    orientation = "正面" if face_visible else "背面"
    
    shot_type_map = {
        "half_body": "半身",
        "full_body": "全身",
        "close_up": "特写",
        "upper_body": "上半身",
        "medium_shot": "中景",
        "long_shot": "远景"
    }
    if shot_type in shot_type_map:
        framing_text = f"转化为标准的{shot_type_map[shot_type]}直立站姿照片。"
    else:
        framing_text = "转化为标准的人像照片，景别与构图可根据原图内容自行合理判断。"

    prompt = (
        f"基于原图人物进行编辑。{framing_text}"
        f"最重要约束：生成的人物必须是{orientation}视角。绝不可旋转人物身体。"
        "保持所有服装、头发和配件细节与原图完美一致。纯白色背景。3:4纵向比例。"
    )
    
    return prompt

def load_orientation_info(image_path):
    """从对应的 JSON 文件中加载人物朝向信息"""
    image_name = image_path.name
    if "_person" in image_name:
        clip_name = image_name.split("_person")[0]
        json_name = f"{clip_name}.json"
        json_path = image_path.parent / json_name
        
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    character_orientations = data.get("character_orientations", {})
                    relative_path = str(image_path.relative_to(INPUT_ROOT))
                    orientation_info = character_orientations.get(relative_path)
                    if orientation_info:
                        return orientation_info.get("face_visible", True), orientation_info.get("shot_type", "full_body")
            except Exception as e:
                pass # 多进程下减少打印噪音
    
    return True, "full_body"

def process_single_image(pipeline, device, input_image_path, output_image_path, prompt):
    try:
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        if output_image_path.exists():
            return True
        
        input_image = Image.open(input_image_path).convert("RGB")
        
        with torch.inference_mode():
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
        print(f"\nError on {device} processing {input_image_path}: {e}")
        return False

def worker(rank, world_size, todo_tasks):
    """
    每个 GPU 进程执行的工作函数
    """
    # 1. 任务分片：当前进程只处理属于自己的那部分任务
    my_tasks = todo_tasks[rank::world_size]
    
    if not my_tasks:
        print(f"[GPU {rank}] No tasks to perform.")
        return

    # 2. 初始化对应显卡上的模型
    pipeline, device = init_pipeline(f"cuda:{rank}")
    
    # 3. 执行推理
    success_count = 0
    # 使用 position=rank 让每个进程的进度条按行整齐排列
    for input_path, output_path, prompt in tqdm(my_tasks, desc=f"GPU {rank}", position=rank):
        if process_single_image(pipeline, device, input_path, output_path, prompt):
            success_count += 1
            
    print(f"\n[GPU {rank}] Finished! Success: {success_count}/{len(my_tasks)}")

def run_manual_mode(input_image_path, output_path=None, face_visible=True, shot_type=None, prompt=None, gpu_id=0):
    """
    人为指定图片和参数进行单张图片编辑。
    
    Args:
        input_image_path: 输入图片路径
        output_path: 输出图片路径，默认在输入同目录下加 _norm 后缀
        face_visible: 人物是否正面朝向
        shot_type: 镜头类型，可为空；为空时由模型自行判断构图
        prompt: 自定义 prompt，若为 None 则根据 face_visible 和 shot_type 自动生成
        gpu_id: 使用的 GPU 编号
    """
    input_path = Path(input_image_path)
    if not input_path.exists():
        print(f"错误：输入图片不存在: {input_path}")
        return False
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_norm{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    if prompt is None:
        prompt = generate_prompt(face_visible, shot_type)
    
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print(f"参数: face_visible={face_visible}, shot_type={shot_type}")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    
    pipeline, device = init_pipeline(f"cuda:{gpu_id}")
    success = process_single_image(pipeline, device, input_path, output_path, prompt)
    if success:
        print(f"处理完成，已保存至: {output_path}")
    return success


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen 图片编辑：批量处理或手动指定单张图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 - 手动模式（单张图片）:
  python qwen_image_edit.py -i path/to/image.jpg
  python qwen_image_edit.py -i image.jpg -o output.jpg --face-visible --shot-type half_body
  python qwen_image_edit.py -i image.jpg --no-face-visible --shot-type full_body --gpu 1

示例 - 批量模式（默认）:
  python qwen_image_edit.py
        """
    )
    parser.add_argument("-i", "--input", type=str, help="手动模式：输入图片路径")
    parser.add_argument("-o", "--output", type=str, help="手动模式：输出图片路径（默认：输入同目录 _norm 后缀）")
    parser.add_argument("--face-visible", dest="face_visible", action="store_true", default=None,
                        help="人物正面朝向（默认）")
    parser.add_argument("--no-face-visible", dest="face_visible", action="store_false",
                        help="人物背面朝向")
    parser.add_argument("--shot-type", type=str, default=None,
                        choices=["half_body", "full_body", "close_up", "upper_body", "medium_shot", "long_shot"],
                        help="镜头类型（默认: 不指定，由模型自行判断）")
    parser.add_argument("--prompt", type=str, default=None, help="自定义 prompt，覆盖自动生成")
    parser.add_argument("--gpu", type=int, default=0, help="手动模式使用的 GPU 编号（默认: 0）")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 手动模式：指定了 -i/--input 时
    if args.input:
        face_visible = args.face_visible if args.face_visible is not None else True
        run_manual_mode(
            input_image_path=args.input,
            output_path=args.output,
            face_visible=face_visible,
            shot_type=args.shot_type,
            prompt=args.prompt,
            gpu_id=args.gpu
        )
        return
    
    # ========== 批量模式 ==========
    # 检测可用的 GPU 数量
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("未检测到可用的 GPU！")
        return
    print(f"检测到 {world_size} 张 GPU，准备启动多进程并行处理...")

    # 1. 主进程扫描所有文件 (避免多进程重复扫描文件系统)
    print("正在扫描文件和生成任务列表...")
    person_images = sorted(list(INPUT_ROOT.rglob("*_person*.jpg")))
    person_images = [img for img in person_images if "_norm" not in img.stem]
    
    todo_tasks = []
    for img_path in person_images:
        output_path = img_path.parent / f"{img_path.stem}_norm{img_path.suffix}"
        if not output_path.exists():
            face_visible, shot_type = load_orientation_info(img_path)
            prompt = generate_prompt(face_visible, shot_type)
            todo_tasks.append((img_path, output_path, prompt))

    total_tasks = len(todo_tasks)
    print(f"扫描完毕。共找到 {total_tasks} 个待处理任务。")
    
    if total_tasks == 0:
        return

    # 2. 设置 multiprocessing 的启动方式为 spawn (CUDA 必须)
    mp.set_start_method('spawn', force=True)

    # 3. 启动多进程
    # nprocs 等于 GPU 数量，自动分配 rank 0 到 world_size-1
    mp.spawn(worker, args=(world_size, todo_tasks), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
