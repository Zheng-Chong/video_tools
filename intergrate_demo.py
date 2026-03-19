import os
import json
import math
import multiprocessing as mp
from tqdm import tqdm

# 这里不需要在全局导入 LightASDPipeline，留到子进程中导入，避免主进程占用显存或引发 CUDA 错误
# from light_asd_pipeline import LightASDPipeline

def bb_intersection_over_face(body_box, face_box):
    """
    计算 IoF (Intersection over Face)
    """
    xA = max(body_box[0], face_box[0])
    yA = max(body_box[1], face_box[1])
    xB = min(body_box[2], face_box[2])
    yB = min(body_box[3], face_box[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    faceArea = max(0, face_box[2] - face_box[0]) * max(0, face_box[3] - face_box[1])
    
    if faceArea == 0:
        return 0.0
    return interArea / float(faceArea)

def process_single_video(pipeline, video_path, json_path, out_json_path, workspace_dir):
    """处理单个视频并保存结果"""
    if not os.path.exists(json_path):
        return False

    # 1. 运行 Light-ASD (使用进程专属的临时目录避免冲突)
    vidTracks, scores = pipeline.process_video(video_path, workspace_base=workspace_dir)
    
    # 2. 读取 Qwen 的全身框数据
    with open(json_path, 'r', encoding='utf-8') as f:
        qwen_data = json.load(f)
        
    characters = qwen_data.get("grounding_info", {}).get("characters", [])
    speaker_results = {}
    
    width, height = 854, 480 

    # 3. 执行身体与面部的匹配 (Late Fusion)
    for char in characters:
        char_id = char.get("person_id")
        bbox_norm = char.get("bbox")
        
        qwen_box = [
            bbox_norm.get("x_min") * width,
            bbox_norm.get("y_min") * height,
            bbox_norm.get("x_max") * width,
            bbox_norm.get("y_max") * height
        ]
        
        best_iof = 0
        best_track_idx = -1
        
        for tidx, track_info in enumerate(vidTracks):
            track_bboxes = track_info['track']['bbox']
            mid_idx = len(track_bboxes) // 2
            asd_face_box = track_bboxes[mid_idx]
            
            iof = bb_intersection_over_face(qwen_box, asd_face_box)
            if iof > best_iof:
                best_iof = iof
                best_track_idx = tidx
                
        if best_track_idx != -1 and best_iof > 0.8: 
            track_scores = scores[best_track_idx]
            avg_score = sum(track_scores) / len(track_scores)
            
            speaker_results[f"<char_{char_id}>"] = {
                "is_speaking": bool(avg_score >= 0),
                "asd_score": round(avg_score, 3)
            }
        else:
            speaker_results[f"<char_{char_id}>"] = {
                "is_speaking": False,
                "asd_score": None,
                "note": "Face missing or occluded"
            }

    # 4. 保存结果到新的 JSON 文件
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(speaker_results, f, indent=4, ensure_ascii=False)
        
    return True

def worker_process(worker_id, gpu_id, tasks):
    """
    子进程执行函数
    worker_id: 进程唯一ID
    gpu_id: 分配给该进程的显卡ID
    tasks: 该进程需要处理的 (video_path, json_path, out_json_path) 列表
    """
    # 1. 物理隔离显卡：让该进程只看得到分配给它的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 2. 在子进程内局部导入，确保模型加载在正确的独立环境中
    from tools.light_asd_pipeline import LightASDPipeline
    
    # 既然已经通过环境变量隔离了，这里 device 统一传 'cuda' 即可
    pipeline = LightASDPipeline(pretrain_model_path="tools/Light-ASD/weight/pretrain_AVA_CVPR.model", device="cuda")
    
    # 3. 为当前进程分配专属的临时工作目录，防止文件覆写冲突
    workspace_dir = f"./asd_temp/worker_{worker_id}"
    os.makedirs(workspace_dir, exist_ok=True)

    # 4. 使用 position 参数将该进程的进度条固定在终端的第 worker_id 行
    for video_path, json_path, out_json_path in tqdm(tasks, desc=f"Worker {worker_id:02d} (GPU {gpu_id})", position=worker_id):
        try:
            process_single_video(pipeline, video_path, json_path, out_json_path, workspace_dir)
        except Exception as e:
            tqdm.write(f"[Worker {worker_id}] 错误 {video_path}: {e}")

def main():
    # ！！！非常重要：PyTorch 多进程必须使用 spawn 模式启动 ！！！
    mp.set_start_method('spawn', force=True)

    # ------------------ 可配参数区 ------------------
    gpus = [0, 1, 2, 3, 4, 5]           # 你拥有的显卡列表 (比如这里用了4张卡)
    instances_per_gpu = 3         # 每张显卡开几个进程实例 (例如 24G 显存开2~3个没问题，按需调整)
    # ------------------------------------------------

    video_base_dir = "/home/chongzheng_p23/data/Projects/video_tools/Datasets/AVAGen-480P"
    json_base_dir = "/home/chongzheng_p23/data/Projects/video_tools/Datasets/AVAGen-480P-Grounded-Caption"
    output_base_dir = "/home/chongzheng_p23/data/Projects/video_tools/Datasets/AVAGen-480P-Speaker-Results"
    
    print("正在扫描数据集目录寻找待处理任务...")
    tasks = []
    for root, dirs, files in os.walk(video_base_dir):
        for file in files:
            if not file.endswith(".mp4"):
                continue
            
            video_path = os.path.join(root, file)
            rel_path = os.path.relpath(video_path, video_base_dir)
            rel_dir = os.path.dirname(rel_path)
            file_name = os.path.splitext(os.path.basename(file))[0]
            
            json_path = os.path.join(json_base_dir, rel_dir, f"{file_name}.json")
            out_json_path = os.path.join(output_base_dir, rel_dir, f"{file_name}_speaker.json")
            
            # 断点续传：过滤掉已经处理完成的文件
            if os.path.exists(out_json_path):
                continue
                
            tasks.append((video_path, json_path, out_json_path))
                
    total_tasks = len(tasks)
    print(f"扫描完毕。共有 {total_tasks} 个视频需要处理。")
    if total_tasks == 0:
        return

    # 计算总进程数并划分任务块 (Chunks)
    total_workers = len(gpus) * instances_per_gpu
    chunk_size = math.ceil(total_tasks / total_workers)
    chunks = [tasks[i:i + chunk_size] for i in range(0, total_tasks, chunk_size)]

    print(f"启动 {total_workers} 个工作进程 (每张卡 {instances_per_gpu} 个实例)...")
    print("\n" * total_workers) # 为每个进度条预留空行空间，防止终端输出混乱
    
    processes = []
    for worker_id in range(total_workers):
        if worker_id < len(chunks):
            # 轮询分配 GPU
            gpu_id = gpus[worker_id % len(gpus)]
            task_chunk = chunks[worker_id]
            
            p = mp.Process(target=worker_process, args=(worker_id, gpu_id, task_chunk))
            processes.append(p)
            p.start()
            
    # 等待所有进程执行完毕
    for p in processes:
        p.join()
        
    print("\n所有任务处理完成！")

if __name__ == "__main__":
    main()