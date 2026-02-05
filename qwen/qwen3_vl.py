import torch
import json
import os
import re
import warnings
import logging
import cv2
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, Manager, Lock, Value
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ================= 配置区域 =================
INPUT_ROOT = Path("Datasets/AVAGen-480P")
OUTPUT_ROOT = Path("Datasets/AVAGen-480P-Character-bbox")
FAILED_LOG_DIR = OUTPUT_ROOT / "_failed_logs"  # 专门存放解析失败的日志
MODEL_ID = "Models/Qwen3-VL-4B-Instruct"
# 每块 GPU 上启动的模型进程数（即每卡多少个模型实例）
PROCESSES_PER_GPU = 2

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

# ================= 1. 视频帧提取函数 =================
def extract_frame_from_video(video_path, output_img_path, frame_index=None):
    """
    从视频中提取一帧并保存为图片
    Args:
        video_path: 视频文件路径
        output_img_path: 输出图片路径
        frame_index: 要提取的帧索引，如果为None则提取中间帧
    Returns:
        成功返回True，失败返回False
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return False
        
        # 如果没有指定帧索引，使用中间帧
        if frame_index is None:
            frame_index = total_frames // 2
        
        # 确保帧索引在有效范围内
        frame_index = min(frame_index, total_frames - 1)
        
        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return False
        
        # 确保输出目录存在
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存图片
        cv2.imwrite(str(output_img_path), frame)
        return True
    except Exception as e:
        print(f"Error extracting frame from {video_path}: {e}")
        return False

# ================= 2. 核心处理函数 =================
def analyze_single_image(image_path):
    """处理单个图片，使用当前进程的模型"""
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
Task: Analyze the image to identify main characters (ignore background pedestrians/people). 
Output Format: ONLY valid JSON. No markdown blocks. No explanations.

Target JSON Structure:
```json
{
    "main_character_count": integer,  // 1 or 2 (only count main characters, ignore background people)
    "characters": [
        {
            "person_id": integer,     // 1 or 2
            "bbox": {
                "x_min": float,       // normalized [0, 1]
                "y_min": float,       // normalized [0, 1]
                "x_max": float,       // normalized [0, 1]
                "y_max": float        // normalized [0, 1]
            }
        }
    ]
}
```

Important:
- Only count main characters who appear prominently in the scene
- Ignore background pedestrians, crowd members, or people in the distance
- If there is only 1 main character, return only 1 entry in characters array
- If there are 2 main characters, return 2 entries
- Bbox coordinates should be normalized to [0, 1] range relative to image dimensions
- Each bbox should tightly cover the WHOLE visible body of the character (not just the face or head).
- If part of the body is outside the image, the bbox should cover all visible body parts (from head to feet/hands as much as possible).
- Do NOT output bbox that only covers the head, face, or shoulders.
- Provide bbox for the main characters visible in this image
"""}
            ],
        },
    ]

    # 准备输入 - 确保 device 上下文正确设置
    # 设置当前 CUDA device，这样 processor 可以正确获取 device
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

# ================= 3. 处理单个文件的包装函数 =================
def process_single_file(args):
    """处理单个文件的包装函数，用于多进程调用"""
    video_file, error_lock = args
    
    try:
        relative_path = video_file.relative_to(INPUT_ROOT)
        output_json_path = OUTPUT_ROOT / relative_path.with_suffix(".json")
        # 图片保存路径：与json文件同目录，同文件名但扩展名为.jpg
        output_img_path = output_json_path.with_suffix(".jpg")
        
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        # --- 断点续传 ---
        if output_json_path.exists() and output_json_path.stat().st_size > 0:
            return {"status": "skipped", "file": video_file.name}

        # --- 提取视频帧并保存为图片 ---
        if not output_img_path.exists():
            if not extract_frame_from_video(video_file, output_img_path):
                return {"status": "error", "file": video_file.name, "error": "Failed to extract frame from video"}

        # --- 执行推理 ---
        raw_result = analyze_single_image(output_img_path)
        parsed_json = clean_and_parse_json(raw_result)

        status = "success"
        if not parsed_json:
            status = "parse_error"
            # 将失败的原始内容保存，方便 debug
            failed_log_path = FAILED_LOG_DIR / (video_file.name + ".txt")
            with open(failed_log_path, "w", encoding="utf-8") as f:
                f.write(f"RAW OUTPUT:\n{raw_result}\n")

        # --- 根据输出裁剪人物并决定是否保留整帧图片 ---
        image_path_rel = None
        character_crops = []
        main_count = 0

        if isinstance(parsed_json, dict):
            main_count = parsed_json.get("main_character_count", 0) or 0

        # 如果有主角，保留整帧图片并按 bbox 裁剪人物
        if main_count > 0 and output_img_path.exists():
            image_path_rel = str(output_img_path.relative_to(OUTPUT_ROOT))

            try:
                img = cv2.imread(str(output_img_path))
                if img is not None and isinstance(parsed_json, dict):
                    h, w = img.shape[:2]
                    characters = parsed_json.get("characters") or []
                    for idx, ch in enumerate(characters):
                        if not isinstance(ch, dict):
                            continue
                        bbox = ch.get("bbox") or {}
                        try:
                            x_min = float(bbox.get("x_min", 0.0))
                            y_min = float(bbox.get("y_min", 0.0))
                            x_max = float(bbox.get("x_max", 0.0))
                            y_max = float(bbox.get("y_max", 0.0))
                        except (TypeError, ValueError):
                            continue

                        # 归一化坐标转为像素，并裁剪到合法范围
                        x_min_px = max(0, min(w - 1, int(x_min * w)))
                        y_min_px = max(0, min(h - 1, int(y_min * h)))
                        x_max_px = max(0, min(w, int(x_max * w)))
                        y_max_px = max(0, min(h, int(y_max * h)))

                        if x_max_px <= x_min_px or y_max_px <= y_min_px:
                            continue

                        crop = img[y_min_px:y_max_px, x_min_px:x_max_px]
                        if crop is None or crop.size == 0:
                            continue

                        person_id = ch.get("person_id") or (idx + 1)
                        crop_path = output_img_path.with_name(
                            f"{output_img_path.stem}_person{person_id}.jpg"
                        )
                        cv2.imwrite(str(crop_path), crop)
                        character_crops.append(str(crop_path.relative_to(OUTPUT_ROOT)))
            except Exception as e:
                print(f"Error cropping characters for {video_file}: {e}")

        # 如果没有主角（0），则删除整帧图片文件（不需要保存）
        if main_count == 0 and output_img_path.exists():
            try:
                output_img_path.unlink()
            except Exception as e:
                print(f"Error removing image {output_img_path}: {e}")

        # --- 保存结果 ---
        final_data = {
            "file_path": str(relative_path),
            "image_path": image_path_rel,
            "character_crops": character_crops,
            "status": status,
            "analysis": parsed_json if parsed_json else {}
        }

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

        return {"status": status, "file": video_file.name}

    except Exception as e:
        # 使用锁保护错误日志写入
        with error_lock:
            with open(OUTPUT_ROOT / "error_log.txt", "a") as ef:
                ef.write(f"{video_file}: {str(e)}\n")
        return {"status": "error", "file": video_file.name, "error": str(e)}

# ================= 4. 批量处理主循环 =================
def main():
    # 检测可用 GPU 数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Warning: No CUDA devices found. Falling back to CPU (not recommended).")
        num_gpus = 1
    else:
        total_procs = num_gpus * PROCESSES_PER_GPU
        print(f"Found {num_gpus} GPU(s). Will use {total_procs} processes "
              f"({PROCESSES_PER_GPU} model workers per GPU) for parallel processing.")
    
    print(f"Scanning for mp4 files in {INPUT_ROOT}...")
    all_files = list(INPUT_ROOT.rglob("*.mp4"))
    print(f"Found {len(all_files)} files.")

    # 过滤已完成的文件
    todo_files = []
    for video_file in all_files:
        relative_path = video_file.relative_to(INPUT_ROOT)
        output_json_path = OUTPUT_ROOT / relative_path.with_suffix(".json")
        if not (output_json_path.exists() and output_json_path.stat().st_size > 0):
            todo_files.append(video_file)
    
    print(f"Found {len(todo_files)} files to process (skipping {len(all_files) - len(todo_files)} already completed).")

    if not todo_files:
        print("All files already processed!")
        return

    # 创建 Manager 用于共享锁
    manager = Manager()
    error_lock = manager.Lock()
    gpu_counter_lock = manager.Lock()  # 保护计数器的锁（通过 Manager 创建以支持进程间共享）
    
    # 使用 multiprocessing.Value 来分配 GPU（Value 可以在进程间共享）
    gpu_counter = Value('i', 0)  # 共享计数器，用于分配 GPU
    
    # 准备参数：每个文件只需要文件路径和错误锁
    task_args = [
        (video_file, error_lock)
        for video_file in todo_files
    ]

    # 创建进程池：每块 GPU 上启动多个进程（多个模型实例）
    gpu_ids = list(range(num_gpus))
    total_procs = num_gpus * PROCESSES_PER_GPU
    with Pool(processes=total_procs, initializer=init_worker, initargs=(gpu_ids, gpu_counter, gpu_counter_lock)) as pool:
        # 使用 imap 以便实时显示进度
        results = list(tqdm(
            pool.imap(process_single_file, task_args),
            total=len(todo_files),
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
            print(f"\n[Warning] {parse_error_count} files had JSON parse errors. Check {FAILED_LOG_DIR} for details.")

if __name__ == "__main__":
    main()

