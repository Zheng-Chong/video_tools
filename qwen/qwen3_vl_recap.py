import torch
import json
import os
import re
import warnings
import logging
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, Manager, Lock, Value
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ================= 配置区域 =================
# 包含原始 Caption 描述的目录 (底稿来源)
CAPTION_ROOT = Path("Datasets/AVAGen-480P")
# 包含图片和已有 Bbox JSON 的目录 (视觉事实与ID来源)
BBOX_ROOT = Path("Datasets/AVAGen-480P-Character-bbox")
# 最终融合后输出的目录
OUTPUT_ROOT = Path("Datasets/AVAGen-480P-Grounded-Caption")

FAILED_LOG_DIR = OUTPUT_ROOT / "_failed_logs"
MODEL_ID = "Models/Qwen3-VL-4B-Instruct"
PROCESSES_PER_GPU = 2

# ================= 0. 屏蔽警告与日志清理 =================
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

_worker_model = None
_worker_processor = None
_worker_gpu_id = None

def init_worker(gpu_ids, counter, counter_lock):
    global _worker_model, _worker_processor, _worker_gpu_id
    with counter_lock:
        worker_index = counter.value
        counter.value += 1
        gpu_id = gpu_ids[worker_index % len(gpu_ids)]
    
    _worker_gpu_id = gpu_id
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    print(f"[Process {os.getpid()}] Loading model on GPU {gpu_id}...")
    _worker_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype="auto", device_map=device
    )
    _worker_processor = AutoProcessor.from_pretrained(MODEL_ID)

FAILED_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ================= 1. 核心处理函数 =================
def generate_grounded_caption(image_path, bbox_data, caption_data):
    """结合图片、已有bbox坐标和原始Caption，生成Grounded Caption"""
    global _worker_model, _worker_processor, _worker_gpu_id
    
    model = _worker_model
    processor = _worker_processor
    device = f"cuda:{_worker_gpu_id}" if _worker_gpu_id is not None else model.device
    
    # 1. 提取原始 Caption 信息作为参考
    caption_ref = caption_data.get("caption", {})
    if not caption_ref:
        reference_text = "No reference available. Describe the scene from scratch."
    else:
        reference_text = json.dumps(caption_ref, ensure_ascii=False)
        
    # 2. 提取并格式化已有的 Bbox 角色信息，告诉模型 ID 和位置的对应关系
    characters_info = ""
    # 适配之前的解析格式：analysis -> characters
    analysis = bbox_data.get("analysis", {})
    char_list = analysis.get("characters", []) if isinstance(analysis, dict) else []
    
    if not char_list:
        characters_info = "No specific main characters detected. Describe the scene generally."
    else:
        for ch in char_list:
            p_id = ch.get("person_id")
            bbox = ch.get("bbox", {})
            # 转换为字符串以便模型理解
            b_str = f"[x_min:{bbox.get('x_min',0):.3f}, y_min:{bbox.get('y_min',0):.3f}, x_max:{bbox.get('x_max',0):.3f}, y_max:{bbox.get('y_max',0):.3f}]"
            characters_info += f"- <char_{p_id}> is located at bounding box {b_str}\n"

    prompt = f"""
Task: You are an expert multimodal video describer. 
I have already identified the main characters and their locations in the image:
{characters_info}

Here is a reference draft description of the scene, which includes camera angles, lighting, scene setting, and character details:
{reference_text}

Instructions:
1. Observe the image carefully. Correct any errors regarding character appearance/actions in the reference draft based on the image.
2. You MUST preserve the environmental details (scene setting, lighting, mood, camera style) from the reference, adjusting them only if they clearly contradict the image.
3. Replace generic character names from the draft (like "the bald man") with the exact tags <char_1>, <char_2>, etc., matching the bounding box locations provided.
4. Output STRICTLY in JSON format.

Target JSON Structure:
{{
    "environment_and_camera": "Describe the scene setting, lighting, mood, and camera details (e.g., 'A static low-angle medium shot in a dimly lit, industrial-looking environment with a tense mood.').",
    "grounded_character_actions": "Describe the spatial layout and what the characters are doing using the ID tags. E.g., '<char_1> is seated on the left gripping a metal bar, looking focused. <char_2> is sitting next to him looking upward...'",
    "full_combined_caption": "Combine the environment and character actions into one rich, flowing paragraph."
}}
"""

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You output strict JSON format only. No markdown."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt}
            ],
        },
    ]

    if _worker_gpu_id is not None:
        torch.cuda.set_device(_worker_gpu_id)
    
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return output_text

def clean_and_parse_json(raw_text):
    try:
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()
        if "assistant" in clean_text.lower():
            idx = clean_text.lower().find("assistant")
            clean_text = clean_text[idx + len("assistant"):].strip()
            if clean_text.startswith(":"):
                clean_text = clean_text[1:].strip()
        return json.loads(clean_text)
    except json.JSONDecodeError:
        try:
            match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except:
            pass
    return None

# ================= 2. 文件处理包装函数 =================
def process_single_file(args):
    bbox_json_path, error_lock = args
    
    try:
        # 相对路径，例如：Alien 3 Special Edition 1992/clip_000029.json
        relative_path = bbox_json_path.relative_to(BBOX_ROOT)
        
        # 1. 确定各个相关文件的路径
        image_path = BBOX_ROOT / relative_path.with_suffix(".jpg")
        caption_json_path = CAPTION_ROOT / relative_path
        output_json_path = OUTPUT_ROOT / relative_path
        
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        # 断点续传
        if output_json_path.exists() and output_json_path.stat().st_size > 0:
            return {"status": "skipped", "file": bbox_json_path.name}

        # 检查依赖文件是否齐全
        if not image_path.exists():
            return {"status": "error", "file": bbox_json_path.name, "error": "Missing image file"}
        if not caption_json_path.exists():
            return {"status": "error", "file": bbox_json_path.name, "error": "Missing original caption json"}

        # 2. 读取两边的数据
        with open(bbox_json_path, 'r', encoding='utf-8') as f:
            bbox_data = json.load(f)
            
        with open(caption_json_path, 'r', encoding='utf-8') as f:
            caption_data = json.load(f)

        # 3. 调用模型生成
        raw_result = generate_grounded_caption(image_path, bbox_data, caption_data)
        parsed_json = clean_and_parse_json(raw_result)

        status = "success"
        if not parsed_json:
            status = "parse_error"
            with open(FAILED_LOG_DIR / (bbox_json_path.name + ".txt"), "w", encoding="utf-8") as f:
                f.write(f"RAW OUTPUT:\n{raw_result}\n")

        # 4. 保存融合后的结果
        final_data = {
            "file_path": str(relative_path),
            "status": status,
            "grounding_info": {
                # 保留原有的人物坐标信息，因为 <char_1> 就对应这里的 person_id: 1
                "characters": bbox_data.get("analysis", {}).get("characters", [])
            },
            "rewritten_result": parsed_json if parsed_json else {}
        }

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

        return {"status": status, "file": bbox_json_path.name}

    except Exception as e:
        with error_lock:
            with open(OUTPUT_ROOT / "error_log.txt", "a") as ef:
                ef.write(f"{bbox_json_path}: {str(e)}\n")
        return {"status": "error", "file": bbox_json_path.name, "error": str(e)}

# ================= 3. 主循环 =================
def main():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("Warning: No CUDA devices found.")
        num_gpus = 1
    total_procs = num_gpus * PROCESSES_PER_GPU
    
    print(f"Scanning for parsed bbox JSONs in {BBOX_ROOT}...")
    # 只扫描 BBOX 目录下的 json 文件
    all_bbox_jsons = list(BBOX_ROOT.rglob("*.json"))
    
    todo_files = []
    for f in all_bbox_jsons:
        rel = f.relative_to(BBOX_ROOT)
        out_path = OUTPUT_ROOT / rel
        if not (out_path.exists() and out_path.stat().st_size > 0):
            todo_files.append(f)
            
    print(f"Found {len(todo_files)} files to process.")
    if not todo_files:
        return

    manager = Manager()
    error_lock = manager.Lock()
    gpu_counter_lock = manager.Lock()
    gpu_counter = Value('i', 0)
    
    task_args = [(f, error_lock) for f in todo_files]
    gpu_ids = list(range(num_gpus))

    with Pool(processes=total_procs, initializer=init_worker, initargs=(gpu_ids, gpu_counter, gpu_counter_lock)) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, task_args),
            total=len(todo_files), desc="Processing"
        ))
        
        success_count = sum(1 for r in results if r["status"] == "success")
        print(f"\nCompleted: Success={success_count}, Errors={len(results)-success_count}")

if __name__ == "__main__":
    main()