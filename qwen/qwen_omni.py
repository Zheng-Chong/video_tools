import torch
import json
import os
import re
import warnings
import logging
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, Manager, Lock, Value
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# ================= 配置区域 =================
INPUT_ROOT = Path("Datasets/MovieLines_Clips_480P_ReCap")
OUTPUT_ROOT = Path("Datasets/MovieLines_Clips_480P_ReCap_Audio_cap")
FAILED_LOG_DIR = OUTPUT_ROOT / "_failed_logs" # 专门存放解析失败的日志
MODEL_ID = "Models/Qwen2.5-Omni-3B"

# ================= 0. 屏蔽警告与日志清理 =================
# 屏蔽 librosa 的 PySoundFile 警告 (因为读取 MP4 音频时这是预期的)
warnings.filterwarnings("ignore", category=UserWarning, module="qwen_omni_utils.v2_5.audio_process")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")

# 屏蔽 Qwen 关于 System Prompt 的警告
logging.getLogger("root").setLevel(logging.ERROR)

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
    _worker_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map=device
    )
    _worker_processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    print(f"[Process {os.getpid()}] Model loaded successfully on GPU {gpu_id}.")

# 确保失败日志目录存在
FAILED_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ================= 2. 核心处理函数 =================
def analyze_single_video(video_path):
    """处理单个视频，使用当前进程的模型"""
    global _worker_model, _worker_processor
    
    model = _worker_model
    processor = _worker_processor
    
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are an expert audio analysis AI. You output strict JSON format only."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(video_path)},
                {"type": "text", "text": """
Task: Analyze the audio track.
Output Format: ONLY valid JSON. No markdown blocks. No explanations.

Target JSON Structure:
```json`
{
    "background_music": boolean,   
    "sound_effects": boolean,   
    "noise": boolean,   
    "multi-speaker": boolean,   
    "speaker_profile": {
        "gender": "string",      // "Male", "Female", "Unknown"
        "age_group": "string",   
        "emotion": "string",     
        "speech_rate": "string", 
        "timbre": "string",      
        "accent": "string",      
        "loudness": "string"     
    }
}
```
"""}
            ],
        },
    ]

    USE_AUDIO_IN_VIDEO = False 
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    # process_mm_info 可能会产生大量 librosa 警告，已被上方代码屏蔽
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    inputs = processor(
        text=text, audio=audios, images=images, videos=videos, 
        return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        text_ids = model.generate(
            **inputs, 
            use_audio_in_video=USE_AUDIO_IN_VIDEO, 
            return_audio=False,
            max_new_tokens=256
        )

    return processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def clean_and_parse_json(raw_text):
    try:
        # 尝试移除 ```json 和 ``` 标记
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()
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
        
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        # --- 断点续传 ---
        if output_json_path.exists() and output_json_path.stat().st_size > 0:
            return {"status": "skipped", "file": video_file.name}

        # --- 执行推理 ---
        raw_result = analyze_single_video(video_file)
        raw_result = raw_result[raw_result.find("assistant")+len("assistant"):]
        parsed_json = clean_and_parse_json(raw_result)

        status = "success"
        if not parsed_json:
            status = "parse_error"
            # 将失败的原始内容保存，方便 debug
            failed_log_path = FAILED_LOG_DIR / (video_file.name + ".txt")
            with open(failed_log_path, "w", encoding="utf-8") as f:
                f.write(f"RAW OUTPUT:\n{raw_result}\n")

        # --- 保存结果 ---
        final_data = {
            "file_path": str(relative_path),
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
        print(f"Found {num_gpus} GPU(s). Will use {num_gpus} processes for parallel processing.")
    
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

    # 创建进程池，每个 GPU 一个进程
    gpu_ids = list(range(num_gpus))
    with Pool(processes=num_gpus, initializer=init_worker, initargs=(gpu_ids, gpu_counter, gpu_counter_lock)) as pool:
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