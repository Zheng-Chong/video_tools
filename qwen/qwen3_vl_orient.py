import torch
import json
import os
import re
import warnings
import logging
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, Manager, Lock, Value
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ================= 配置区域 =================
INPUT_ROOT = Path("Datasets/AVAGen-480P-Character-bbox")  # 包含裁剪后人物图像的目录
MODEL_ID = "Models/Qwen3-VL-4B-Instruct"
# 每块 GPU 上启动的模型进程数（即每卡多少个模型实例）
PROCESSES_PER_GPU = 2
FAILED_LOG_DIR = INPUT_ROOT / "_orientation_failed_logs"  # 专门存放解析失败的日志

# ================= 0. 屏蔽警告与日志清理 =================
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# 全局变量用于存储每个进程的模型实例
_worker_model = None
_worker_processor = None
_worker_gpu_id = None

def init_worker(gpu_ids, counter, counter_lock):
    """初始化工作进程，为每个进程加载模型到指定的 GPU"""
    global _worker_model, _worker_processor, _worker_gpu_id
    
    # 使用原子计数器来分配 GPU，确保每个进程获得不同的 GPU
    with counter_lock:
        worker_index = counter.value
        counter.value += 1
        gpu_id = gpu_ids[worker_index % len(gpu_ids)]
    
    _worker_gpu_id = gpu_id
    
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    print(f"[Process {os.getpid()}] Loading model on GPU {gpu_id} from {MODEL_ID}...")
    _worker_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype="auto", device_map=device
    )
    _worker_processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"[Process {os.getpid()}] Model loaded successfully on GPU {gpu_id}.")

# 确保失败日志目录存在
FAILED_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ================= 1. 核心分析函数 =================
def analyze_character_orientation(image_path):
    """分析人物图像的面部可见性和拍摄类型"""
    global _worker_model, _worker_processor, _worker_gpu_id
    
    model = _worker_model
    processor = _worker_processor
    device = f"cuda:{_worker_gpu_id}" if _worker_gpu_id is not None else model.device
    
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are an expert image analysis AI. You output strict JSON format only."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": """
Task: Analyze the character image to determine face visibility and shot type.
Output Format: ONLY valid JSON. No markdown blocks. No explanations.

Target JSON Structure:
```json
{
    "face_visible": boolean,        // true if face is clearly visible, false otherwise
    "shot_type": string             // "full_body" (全身), "half_body" (半身), or "close_up" (大头照)
}
```

Important:
- "face_visible": true only if the face (especially eyes, nose, mouth) is clearly visible and not occluded, blurred, or turned away
- "shot_type" classification:
  * "full_body": The entire body from head to feet is visible (or at least from head to below waist/knees)
  * "half_body": Only upper body is visible (from head to waist/chest, but not full body)
  * "close_up": Only head, face, or upper shoulders are visible (headshot or portrait style)
- If the person is partially outside the image frame, classify based on what is visible
- Be strict: only mark face_visible as true if the face features are clearly discernible
"""}
            ],
        },
    ]

    # 准备输入 - 确保 device 上下文正确设置
    if _worker_gpu_id is not None:
        torch.cuda.set_device(_worker_gpu_id)
    
    # 准备输入
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    # 确保所有 tensor 都在正确的 device 上
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # 推理
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return output_text

def clean_and_parse_json(raw_text):
    """清理并解析 JSON 输出"""
    try:
        # 尝试移除 ```json 和 ``` 标记
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()
        # 移除可能的 "assistant" 前缀
        if "assistant" in clean_text.lower():
            idx = clean_text.lower().find("assistant")
            clean_text = clean_text[idx + len("assistant"):].strip()
            # 移除可能的冒号
            if clean_text.startswith(":"):
                clean_text = clean_text[1:].strip()
        return json.loads(clean_text)
    except json.JSONDecodeError:
        # 正则提取：寻找最外层的 {}
        try:
            match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except:
            pass
    return None

# ================= 2. 处理单个文件的包装函数 =================
def process_single_image(args):
    """处理单个人物图像，用于多进程调用"""
    image_path, error_lock = args
    
    try:
        # 对应的 JSON 文件路径（与原视频的 JSON 文件相同）
        # 图像路径格式：.../clip_XXXXX_personN.jpg
        # 对应的 JSON 路径：.../clip_XXXXX.json
        image_stem = image_path.stem
        # 移除 _personN 后缀，得到原始文件名
        base_stem = re.sub(r'_person\d+$', '', image_stem)
        json_path = image_path.parent / f"{base_stem}.json"
        
        # 如果对应的 JSON 文件不存在，跳过（说明原视频还没处理完）
        if not json_path.exists():
            return {"status": "skipped", "file": image_path.name, "reason": "corresponding_json_not_found"}
        
        # 读取现有的 JSON 文件
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except Exception as e:
            return {"status": "error", "file": image_path.name, "error": f"Failed to read JSON: {str(e)}"}
        
        # 检查是否已经处理过（如果已有 orientation 信息）
        image_rel_path = str(image_path.relative_to(INPUT_ROOT))
        if "character_orientations" in existing_data:
            orientations = existing_data.get("character_orientations", {})
            if image_rel_path in orientations:
                return {"status": "skipped", "file": image_path.name, "reason": "already_processed"}
        
        # --- 执行推理 ---
        raw_result = analyze_character_orientation(image_path)
        parsed_json = clean_and_parse_json(raw_result)

        status = "success"
        if not parsed_json:
            status = "parse_error"
            # 将失败的原始内容保存，方便 debug
            failed_log_path = FAILED_LOG_DIR / (image_path.name + ".txt")
            with open(failed_log_path, "w", encoding="utf-8") as f:
                f.write(f"RAW OUTPUT:\n{raw_result}\n")
        
        # --- 更新 JSON 文件 ---
        if status == "success" and parsed_json:
            # 初始化 character_orientations 字段
            if "character_orientations" not in existing_data:
                existing_data["character_orientations"] = {}
            
            # 保存该图像的标注信息
            existing_data["character_orientations"][image_rel_path] = {
                "face_visible": parsed_json.get("face_visible", False),
                "shot_type": parsed_json.get("shot_type", "unknown")
            }
            
            # 写回 JSON 文件
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False)
        elif status == "parse_error":
            # 即使解析失败，也记录一下
            if "character_orientations" not in existing_data:
                existing_data["character_orientations"] = {}
            existing_data["character_orientations"][image_rel_path] = {
                "face_visible": None,
                "shot_type": None,
                "status": "parse_error"
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False)

        return {"status": status, "file": image_path.name}

    except Exception as e:
        # 使用锁保护错误日志写入
        with error_lock:
            with open(INPUT_ROOT / "orientation_error_log.txt", "a") as ef:
                ef.write(f"{image_path}: {str(e)}\n")
        return {"status": "error", "file": image_path.name, "error": str(e)}

# ================= 3. 批量处理主循环 =================
def main():
    # 确保 multiprocessing 使用 'spawn' 启动方法（如果还没设置）
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过，忽略错误
        pass
    
    # 检测可用 GPU 数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Warning: No CUDA devices found. Falling back to CPU (not recommended).")
        num_gpus = 1
    else:
        total_procs = num_gpus * PROCESSES_PER_GPU
        print(f"Found {num_gpus} GPU(s). Will use {total_procs} processes "
              f"({PROCESSES_PER_GPU} model workers per GPU) for parallel processing.")
    
    print(f"Scanning for character crop images (*_person*.jpg) in {INPUT_ROOT}...")
    # 查找所有人物裁剪图像（格式：*_person*.jpg），并去掉尾缀是 norm 的
    all_images = []
    for pattern in ["*_person*.jpg", "*_person*.JPG", "*_person*.jpeg", "*_person*.JPEG"]:
        all_images.extend(list(INPUT_ROOT.rglob(pattern)))
    # 过滤掉尾缀是 norm 的图像文件
    all_images = [img for img in all_images if "_norm" not in img.stem]
    print(f"Found {len(all_images)} character crop images.")

    # 过滤已完成的文件（通过检查对应的 JSON 文件是否已有该图像的标注）
    todo_images = []
    for image_path in all_images:
        image_stem = image_path.stem
        base_stem = re.sub(r'_person\d+$', '', image_stem)
        json_path = image_path.parent / f"{base_stem}.json"
        
        if not json_path.exists():
            continue  # 跳过没有对应 JSON 的图像
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            image_rel_path = str(image_path.relative_to(INPUT_ROOT))
            
            # 检查是否已处理
            orientations = existing_data.get("character_orientations", {})
            if image_rel_path not in orientations:
                todo_images.append(image_path)
        except:
            # 如果读取失败，也加入待处理列表
            todo_images.append(image_path)
    
    print(f"Found {len(todo_images)} images to process (skipping {len(all_images) - len(todo_images)} already completed).")

    if not todo_images:
        print("All images already processed!")
        return

    # 创建 Manager 用于共享锁
    manager = Manager()
    error_lock = manager.Lock()
    gpu_counter_lock = manager.Lock()
    
    # 使用 multiprocessing.Value 来分配 GPU
    gpu_counter = Value('i', 0)
    
    # 准备参数：每个图像只需要图像路径和错误锁
    task_args = [
        (image_path, error_lock)
        for image_path in todo_images
    ]

    # 创建进程池：每块 GPU 上启动多个进程（多个模型实例）
    gpu_ids = list(range(num_gpus))
    total_procs = num_gpus * PROCESSES_PER_GPU
    with Pool(processes=total_procs, initializer=init_worker, initargs=(gpu_ids, gpu_counter, gpu_counter_lock)) as pool:
        # 使用 imap 以便实时显示进度
        results = list(tqdm(
            pool.imap(process_single_image, task_args),
            total=len(todo_images),
            desc="Processing",
            mininterval=1.0
        ))
        
        # 统计结果
        success_count = sum(1 for r in results if r["status"] == "success")
        parse_error_count = sum(1 for r in results if r["status"] == "parse_error")
        error_count = sum(1 for r in results if r["status"] == "error")
        skipped_count = sum(1 for r in results if r["status"] == "skipped")
        
        print(f"\nProcessing completed!")
        print(f"  Success: {success_count}")
        print(f"  Parse errors: {parse_error_count}")
        print(f"  Errors: {error_count}")
        print(f"  Skipped: {skipped_count}")
        
        # 显示解析错误的文件
        if parse_error_count > 0:
            parse_error_files = [r["file"] for r in results if r["status"] == "parse_error"]
            print(f"\n[Warning] {parse_error_count} images had JSON parse errors. Check {FAILED_LOG_DIR} for details.")

if __name__ == "__main__":
    main()

