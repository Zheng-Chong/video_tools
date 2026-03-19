import os
import time
import json
import threading
import queue
from tools.transnet_pipeline import TransNetV2Pipeline
from tools.person_detection_pipeline import PersonDetectionPipeline
from tools.ocr_pipeline import EasyOCRCreditDetectorPipeline
from tools.whisperx_pipeline import MovieTranscriber
from tools.light_asd_pipeline import LightASDPipeline, align_subtitles_with_speakers
from tools.clearervoice_pipeline import ClearVoicePipeline
from qwen.qwen_image_edit import QwenImageEditRunner

import logging

# ==========================================
# 日志屏蔽
# ==========================================
logging.getLogger("tools.whisperx_pipeline").setLevel(logging.WARNING)
logging.getLogger("pyscenedetect").setLevel(logging.CRITICAL)
logging.getLogger("tools.clearervoice_pipeline").setLevel(logging.WARNING)

# ==========================================
# 配置路径与全局锁
# ==========================================
MOVIE_DIR = "Datasets/Movies"
OUTPUT_DIR = "Datasets/AVAGen"

# ASD 多卡并行：每个 device 启动一个消费者线程，共享同一个任务队列
ASD_DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
IMAGE_EDIT_DEVICES = ["cuda:1", "cuda:2", "cuda:3"]
IMAGE_EDIT_DEFAULT_FACE_VISIBLE = True
IMAGE_EDIT_DEFAULT_SHOT_TYPE = None

# 用于 Clips Check 阶段写入 jsonl 时的文件锁，防止多线程乱序
jsonl_lock = threading.Lock()

# ==========================================
# 工具函数：状态检查
# ==========================================
def _safe_qsize(q):
    try:
        return q.qsize()
    except (NotImplementedError, AttributeError):
        return -1


def _get_completed_counts():
    """统计各阶段已完成的样本数（扫描 OUTPUT_DIR）"""
    counts = {"transnet": 0, "clips": 0, "whisper": 0, "asd": 0, "clearvoice": 0, "portrait_norm": 0}
    try:
        if not os.path.isdir(OUTPUT_DIR):
            return counts
        for mb in os.listdir(OUTPUT_DIR):
            movie_dir = os.path.join(OUTPUT_DIR, mb)
            if not os.path.isdir(movie_dir):
                continue
            if is_transnet_completed(mb):
                counts["transnet"] += 1
            jsonl_file = os.path.join(movie_dir, "clips_check.jsonl")
            if os.path.exists(jsonl_file):
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    counts["clips"] += sum(1 for line in f if line.strip())
            transcripts_dir = os.path.join(movie_dir, "subtitles")
            if os.path.isdir(transcripts_dir):
                counts["whisper"] += sum(1 for f in os.listdir(transcripts_dir) if f.endswith(".json"))
            asd_dir = os.path.join(movie_dir, "asd")
            if os.path.isdir(asd_dir):
                counts["asd"] += sum(
                    1 for f in os.listdir(asd_dir)
                    if f.endswith("_asd_done.txt") and os.path.isfile(os.path.join(asd_dir, f)))
            enhanced_dir = os.path.join(movie_dir, "enhanced_shots")
            if os.path.isdir(enhanced_dir):
                counts["clearvoice"] += sum(1 for f in os.listdir(enhanced_dir) if os.path.isfile(os.path.join(enhanced_dir, f)))
            if os.path.isdir(asd_dir):
                for clip_name in os.listdir(asd_dir):
                    portraits_dir = os.path.join(asd_dir, clip_name, "portraits_yolo")
                    if os.path.isdir(portraits_dir):
                        counts["portrait_norm"] += sum(
                            1 for f in os.listdir(portraits_dir)
                            if f.endswith("_norm.jpg") and os.path.isfile(os.path.join(portraits_dir, f))
                        )
    except Exception:
        pass
    return counts


def _format_all_queue_sizes(all_queues):
    if not all_queues:
        return ""
    try:
        completed = _get_completed_counts()
        stage_order = ["transnet", "clips", "whisper", "asd", "clearvoice", "portrait_norm"]
        pending_items = []
        done_items = []
        for k in stage_order:
            if k in all_queues:
                qs = _safe_qsize(all_queues[k])
                pending_items.append(f"{k}={qs}")
                done_items.append(f"{k}={completed.get(k, 0)}")
        for k in sorted(all_queues.keys()):
            if k not in set(stage_order):
                pending_items.append(f"{k}={_safe_qsize(all_queues[k])}")
                done_items.append(f"{k}={completed.get(k, 0)}")
        parts = []
        if pending_items:
            parts.append("待处理: " + ", ".join(pending_items))
        if done_items:
            parts.append("已完成: " + ", ".join(done_items))
        return " | " + " | ".join(parts) if parts else ""
    except Exception:
        return ""

def is_clip_long_enough_by_name(filename: str, min_seconds: float = 2.0, fps: int = 24) -> bool:
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    if len(parts) < 3:
        return True
    try:
        start_frame, end_frame = int(parts[-2]), int(parts[-1])
        if end_frame <= start_frame:
            return False
        return (end_frame - start_frame) >= int(min_seconds * fps)
    except Exception:
        return True

def is_transnet_completed(movie_basename):
    list_file = os.path.join(OUTPUT_DIR, movie_basename, "shots_list.txt")
    if not os.path.exists(list_file): return False
    try:
        with open(list_file, "r", encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if line]
            if lines and lines[-1].strip() == "# COMPLETE": return True
    except Exception: pass
    return False

def is_stage_completed(movie_basename, marker_name):
    return os.path.exists(os.path.join(OUTPUT_DIR, movie_basename, marker_name))

# ==========================================
# 线程组 1：TransNet (电影级)
# ==========================================
def transnet_producer(task_queue, processing_set, set_lock, all_queues=None):
    print("[TransNet 生产者] 已启动...")
    os.makedirs(MOVIE_DIR, exist_ok=True)
    while True:
        try:
            movies = [f for f in os.listdir(MOVIE_DIR) if f.lower().endswith('.mp4')]
            for movie in movies:
                mb = os.path.splitext(movie)[0]
                with set_lock:
                    if movie in processing_set: continue
                if not is_transnet_completed(mb):
                    with set_lock: processing_set.add(movie)
                    task_queue.put(movie)
        except Exception as e: print(f"[TransNet 生产者] 异常: {e}")
        print("[TransNet 生产者] 轮询完成" + _format_all_queue_sizes(all_queues))
        time.sleep(10) # 缩短轮询时间

def transnet_consumer(task_queue, processing_set, set_lock, transnet_pipeline):
    print("[TransNet 消费者] 已启动...")
    while True:
        try: movie = task_queue.get(timeout=3)
        except queue.Empty: continue
            
        movie_path = os.path.join(MOVIE_DIR, movie)
        mb = os.path.splitext(movie)[0]
        out_dir = os.path.join(OUTPUT_DIR, mb)
        try:
            print(f"[TransNet] 开始切分: {movie}")
            transnet_pipeline.read_video(movie_path)
            transnet_pipeline.shot_detect(output_dir=out_dir, save_mp4=True)
        except Exception as e:
            print(f"[TransNet 消费者] 处理失败: {e}")
        finally:
            with set_lock: processing_set.discard(movie)
            task_queue.task_done()

# ==========================================
# 线程组 2：Clips Check (切片级)
# ==========================================
def clips_check_producer(task_queue, processing_set, set_lock, all_queues=None):
    print("[Clips 生产者] 已启动...")
    while True:
        try:
            for mb in os.listdir(OUTPUT_DIR):
                movie_dir = os.path.join(OUTPUT_DIR, mb)
                if not os.path.isdir(movie_dir) or is_stage_completed(mb, "clips_check_done.txt"):
                    continue

                shots_dir = os.path.join(movie_dir, "shots")
                if not os.path.exists(shots_dir): continue

                mp4_files = [f for f in os.listdir(shots_dir) if f.endswith('.mp4')]
                
                # 读取已处理记录
                processed = set()
                jsonl_file = os.path.join(movie_dir, "clips_check.jsonl")
                if os.path.exists(jsonl_file):
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip(): processed.add(json.loads(line)["file"])

                pending = [f for f in mp4_files if f not in processed]
                for mp4 in pending:
                    item_id = f"{mb}/{mp4}"
                    with set_lock:
                        if item_id not in processing_set:
                            processing_set.add(item_id)
                            task_queue.put((mb, mp4))

                # 检查是否可宣告整部电影结束
                if is_transnet_completed(mb) and len(pending) == 0:
                    in_progress = False
                    with set_lock:
                        for mp4 in mp4_files:
                            if f"{mb}/{mp4}" in processing_set: in_progress = True; break
                    if not in_progress:
                        with open(os.path.join(movie_dir, "clips_check_done.txt"), "w") as f: f.write("COMPLETE")
        except Exception as e: print(f"[Clips 生产者] 异常: {e}")
        print("[Clips 生产者] 轮询完成" + _format_all_queue_sizes(all_queues))
        time.sleep(5)

def clips_check_consumer(task_queue, processing_set, set_lock, ocr_pipe, person_pipe):
    print("[Clips 消费者] 已启动...")
    while True:
        try: mb, mp4_name = task_queue.get(timeout=3)
        except queue.Empty: continue
            
        clip_path = os.path.join(OUTPUT_DIR, mb, "shots", mp4_name)
        jsonl_file = os.path.join(OUTPUT_DIR, mb, "clips_check.jsonl")
        
        try:
            has_text, _ = ocr_pipe.process_clip(clip_path)
        except Exception: has_text = False
            
        try:
            has_person, _ = person_pipe.process_clip(clip_path)
        except Exception: has_person = False
        
        with jsonl_lock:
            with open(jsonl_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"file": mp4_name, "has_text": has_text, "has_person": has_person}) + "\n")
                
        print(f"[Clips 检测] {mb} -> {mp4_name} (Text:{has_text}, Person:{has_person})")
        with set_lock: processing_set.discard(f"{mb}/{mp4_name}")
        task_queue.task_done()

# ==========================================
# 线程组 3：WhisperX (切片级)
# ==========================================
def whisperx_producer(task_queue, processing_set, set_lock, all_queues=None):
    print("[WhisperX 生产者] 已启动...")
    while True:
        try:
            for mb in os.listdir(OUTPUT_DIR):
                movie_dir = os.path.join(OUTPUT_DIR, mb)
                if not os.path.isdir(movie_dir) or is_stage_completed(mb, "transcription_done.txt"):
                    continue

                jsonl_file = os.path.join(movie_dir, "clips_check.jsonl")
                transcripts_dir = os.path.join(movie_dir, "subtitles")
                os.makedirs(transcripts_dir, exist_ok=True)

                valid_clips = []
                if os.path.exists(jsonl_file):
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line.strip())
                            if data.get("has_person", False) and not data.get("has_text", True):
                                if is_clip_long_enough_by_name(data["file"]):
                                    valid_clips.append(data["file"])

                pending = []
                for clip in valid_clips:
                    clip_base = os.path.splitext(clip)[0]
                    if not os.path.exists(os.path.join(transcripts_dir, f"{clip_base}.json")):
                        pending.append(clip)

                for mp4 in pending:
                    item_id = f"{mb}/{mp4}"
                    with set_lock:
                        if item_id not in processing_set:
                            processing_set.add(item_id)
                            task_queue.put((mb, mp4))

                # 判断收尾
                if is_stage_completed(mb, "clips_check_done.txt") and len(pending) == 0:
                    in_progress = False
                    with set_lock:
                        for mp4 in valid_clips:
                            if f"{mb}/{mp4}" in processing_set: in_progress = True; break
                    if not in_progress:
                        with open(os.path.join(movie_dir, "transcription_done.txt"), "w") as f: f.write("COMPLETE")
        except Exception as e: print(f"[Whisper 生产者] 异常: {e}")
        print("[WhisperX 生产者] 轮询完成" + _format_all_queue_sizes(all_queues))
        time.sleep(5)

def whisperx_consumer(task_queue, processing_set, set_lock, transcriber):
    print("[WhisperX 消费者] 已启动...")
    while True:
        try: mb, mp4_name = task_queue.get(timeout=3)
        except queue.Empty: continue
            
        clip_path = os.path.join(OUTPUT_DIR, mb, "shots", mp4_name)
        clip_base = os.path.splitext(mp4_name)[0]
        out_prefix = os.path.join(OUTPUT_DIR, mb, "subtitles", clip_base)
        
        try:
            result = transcriber.process(input_path=clip_path, batch_size=8)
            with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            transcriber.save_srt(result, f"{out_prefix}.srt")
            print(f"[Whisper 转录] {mb} -> {mp4_name} 成功")
        except Exception as e:
            with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
                json.dump({"segments": []}, f)
            print(f"[Whisper 转录] {mb} -> {mp4_name} 失败或无语音")
        finally:
            with set_lock: processing_set.discard(f"{mb}/{mp4_name}")
            task_queue.task_done()

# ==========================================
# 线程组 4：Light ASD (切片级)
# ==========================================
def asd_producer(task_queue, processing_set, set_lock, all_queues=None):
    print("[ASD 生产者] 已启动...")
    while True:
        try:
            for mb in os.listdir(OUTPUT_DIR):
                movie_dir = os.path.join(OUTPUT_DIR, mb)
                if not os.path.isdir(movie_dir) or is_stage_completed(mb, "asd_done.txt"):
                    continue

                transcripts_dir = os.path.join(movie_dir, "subtitles")
                asd_dir = os.path.join(movie_dir, "asd")
                os.makedirs(asd_dir, exist_ok=True)

                dialogue_clips = []
                shots_dir = os.path.join(movie_dir, "shots")
                if os.path.exists(transcripts_dir):
                    for fname in os.listdir(transcripts_dir):
                        if fname.endswith('.json'):
                            try:
                                clip_name = f"{os.path.splitext(fname)[0]}.mp4"
                                if not os.path.isfile(os.path.join(shots_dir, clip_name)):
                                    continue
                                with open(os.path.join(transcripts_dir, fname), "r", encoding="utf-8") as f:
                                    if len(json.load(f).get("segments", [])) > 0:
                                        if is_clip_long_enough_by_name(clip_name):
                                            dialogue_clips.append(clip_name)
                            except Exception: pass

                pending = []
                for clip in dialogue_clips:
                    clip_base = os.path.splitext(clip)[0]
                    if not os.path.exists(os.path.join(asd_dir, f"{clip_base}_asd_done.txt")):
                        pending.append(clip)

                for mp4 in pending:
                    item_id = f"{mb}/{mp4}"
                    with set_lock:
                        if item_id not in processing_set:
                            processing_set.add(item_id)
                            task_queue.put((mb, mp4))

                if is_stage_completed(mb, "transcription_done.txt") and len(pending) == 0:
                    in_progress = False
                    with set_lock:
                        for mp4 in dialogue_clips:
                            if f"{mb}/{mp4}" in processing_set: in_progress = True; break
                    if not in_progress:
                        with open(os.path.join(movie_dir, "asd_done.txt"), "w") as f: f.write("COMPLETE")
        except Exception as e: print(f"[ASD 生产者] 异常: {e}")
        print("[ASD 生产者] 轮询完成" + _format_all_queue_sizes(all_queues))
        time.sleep(5)

def asd_consumer(task_queue, processing_set, set_lock, asd_pipeline, worker_id=0):
    tag = f"[ASD 消费者-{worker_id} ({asd_pipeline.device})]"
    print(f"{tag} 已启动...")
    while True:
        try: mb, mp4_name = task_queue.get(timeout=3)
        except queue.Empty: continue
            
        clip_path = os.path.join(OUTPUT_DIR, mb, "shots", mp4_name)
        clip_base = os.path.splitext(mp4_name)[0]
        asd_dir = os.path.join(OUTPUT_DIR, mb, "asd")
        clip_workspace_dir = os.path.join(asd_dir, clip_base)
        transcripts_dir = os.path.join(OUTPUT_DIR, mb, "subtitles")
        
        try:
            aligned = []
            vid_tracks, scores = asd_pipeline.process_video(clip_path, workspace_base=asd_dir, threshold=-0.4)
            asd_pipeline.export_speaker_portraits_yolo(clip_path, vid_tracks, os.path.join(clip_workspace_dir, "portraits_yolo"))
            
            json_path = os.path.join(transcripts_dir, f"{clip_base}.json")
            srt_path = os.path.join(transcripts_dir, f"{clip_base}.srt")
            
            if os.path.exists(json_path) and os.path.exists(srt_path):
                aligned = align_subtitles_with_speakers(vid_tracks, scores, srt_path, json_path, fps=25, threshold=-0.4)
                res_dir = os.path.join(clip_workspace_dir, "results")
                os.makedirs(res_dir, exist_ok=True)
                with open(os.path.join(res_dir, "person_subtitle_mapping.json"), "w", encoding="utf-8") as f:
                    json.dump(aligned, f, ensure_ascii=False, indent=2)

            if aligned and asd_pipeline.last_track_mp4_paths:
                pytracks_dir = os.path.join(clip_workspace_dir, "pytracks")
                out_mp4 = os.path.join(pytracks_dir, "all_tracks_with_dialog.mp4")
                all_tracks_mp4 = asd_pipeline.last_track_mp4_paths[0]
                audio_file_path = os.path.join(clip_workspace_dir, "pyavi", "audio.wav")
                asd_pipeline.render_aligned_subtitles_on_video(
                    input_video_path=all_tracks_mp4,
                    audio_file_path=audio_file_path,
                    aligned_subtitles=aligned,
                    output_mp4_path=out_mp4,
                    fps=25,
                )

            with open(os.path.join(asd_dir, f"{clip_base}_asd_done.txt"), "w") as f: f.write("DONE")
            print(f"{tag} {mb} -> {mp4_name} 完成")
        except Exception as e:
            with open(os.path.join(asd_dir, f"{clip_base}_asd_done.txt"), "w") as f: f.write("FAILED")
            print(f"{tag} {mb} -> {mp4_name} 失败: {e}")
        finally:
            with set_lock: processing_set.discard(f"{mb}/{mp4_name}")
            task_queue.task_done()

# ==========================================
# 线程组 5：ClearVoice (切片级)
# ==========================================
def clearvoice_producer(task_queue, processing_set, set_lock, all_queues=None):
    print("[ClearVoice 生产者] 已启动...")
    while True:
        try:
            for mb in os.listdir(OUTPUT_DIR):
                movie_dir = os.path.join(OUTPUT_DIR, mb)
                if not os.path.isdir(movie_dir) or is_stage_completed(mb, "clearvoice_done.txt"):
                    continue

                asd_dir = os.path.join(movie_dir, "asd")
                enhanced_dir = os.path.join(movie_dir, "enhanced_shots")
                os.makedirs(enhanced_dir, exist_ok=True)

                # 统计总的 asd 数，UNK 数和通过数
                total_asd = 0
                unk_count = 0
                passed_count = 0
                valid_clips = []
                if os.path.exists(asd_dir):
                    for cb in os.listdir(asd_dir):
                        mapping_file = os.path.join(asd_dir, cb, "results", "person_subtitle_mapping.json")
                        if os.path.isfile(mapping_file):
                            total_asd += 1
                            try:
                                with open(mapping_file, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                has_unk = any("UNK" in str(item.get("speaker", "")).upper() for item in data)
                                if has_unk:
                                    unk_count += 1
                                if not has_unk and len(data) > 0:
                                    valid_clips.append(f"{cb}.mp4")
                                    passed_count += 1
                            except:
                                pass
                # print(f"[ClearVoice] [{mb}]总 ASD 片段数: {total_asd}, 含 UNK 片段数: {unk_count}, 通过片段数: {passed_count}")
                pending = [c for c in valid_clips if not os.path.exists(os.path.join(enhanced_dir, c))]
                
                for mp4 in pending:
                    item_id = f"{mb}/{mp4}"
                    with set_lock:
                        if item_id not in processing_set:
                            processing_set.add(item_id)
                            task_queue.put((mb, mp4))

                if is_stage_completed(mb, "asd_done.txt") and len(pending) == 0:
                    in_progress = False
                    with set_lock:
                        for mp4 in valid_clips:
                            if f"{mb}/{mp4}" in processing_set: in_progress = True; break
                    if not in_progress:
                        with open(os.path.join(movie_dir, "clearvoice_done.txt"), "w") as f: f.write("COMPLETE")
        except Exception as e: print(f"[CV 生产者] 异常: {e}")
        print("[ClearVoice 生产者] 轮询完成" + _format_all_queue_sizes(all_queues))
        time.sleep(5)

def clearvoice_consumer(task_queue, processing_set, set_lock, cv_pipe):
    print("[ClearVoice 消费者] 已启动...")
    while True:
        try:
            mb, mp4_name = task_queue.get(timeout=3)
        except queue.Empty:
            continue

        in_path = os.path.join(OUTPUT_DIR, mb, "shots", mp4_name)
        out_path = os.path.join(OUTPUT_DIR, mb, "enhanced_shots", mp4_name)
        
        try:
            cv_pipe.enhance(in_path, out_path)
            print(f"[ClearVoice 增强] {mb} -> {mp4_name} 完成")
        except Exception as e:
            # 即使失败也触碰一个空文件防止无限重试
            open(out_path, 'a').close()
            print(f"[ClearVoice 增强] {mb} -> {mp4_name} 异常/失败，异常信息: {repr(e)}")
        finally:
            with set_lock:
                processing_set.discard(f"{mb}/{mp4_name}")
            task_queue.task_done()

# ==========================================
# 线程组 6：人物标准化 (人像级)
# ==========================================
def portrait_norm_producer(task_queue, processing_set, set_lock, all_queues=None):
    print("[人物标准化 生产者] 已启动...")
    while True:
        try:
            for mb in os.listdir(OUTPUT_DIR):
                movie_dir = os.path.join(OUTPUT_DIR, mb)
                if not os.path.isdir(movie_dir) or is_stage_completed(mb, "portrait_norm_done.txt"):
                    continue

                asd_dir = os.path.join(movie_dir, "asd")
                enhanced_dir = os.path.join(movie_dir, "enhanced_shots")
                if not os.path.isdir(asd_dir) or not os.path.isdir(enhanced_dir):
                    continue

                pending = []
                clearvoice_clips = {
                    os.path.splitext(f)[0]
                    for f in os.listdir(enhanced_dir)
                    if f.lower().endswith(".mp4") and os.path.isfile(os.path.join(enhanced_dir, f))
                }
                for clip_base in sorted(clearvoice_clips):
                    portraits_dir = os.path.join(asd_dir, clip_base, "portraits_yolo")
                    if not os.path.isdir(portraits_dir):
                        continue
                    for fname in sorted(os.listdir(portraits_dir)):
                        if not fname.lower().endswith(".jpg"):
                            continue
                        if fname.endswith("_norm.jpg"):
                            continue
                        in_path = os.path.join(portraits_dir, fname)
                        out_path = os.path.join(portraits_dir, f"{os.path.splitext(fname)[0]}_norm.jpg")
                        if not os.path.isfile(in_path) or os.path.exists(out_path):
                            continue
                        pending.append((mb, clip_base, fname))

                for item in pending:
                    item_id = "/".join(item)
                    with set_lock:
                        if item_id not in processing_set:
                            processing_set.add(item_id)
                            task_queue.put(item)

                if is_stage_completed(mb, "clearvoice_done.txt") and len(pending) == 0:
                    in_progress = False
                    with set_lock:
                        for item in list(processing_set):
                            if item.startswith(f"{mb}/"):
                                in_progress = True
                                break
                    if not in_progress:
                        with open(os.path.join(movie_dir, "portrait_norm_done.txt"), "w") as f:
                            f.write("COMPLETE")
        except Exception as e:
            print(f"[人物标准化 生产者] 异常: {e}")
        print("[人物标准化 生产者] 轮询完成" + _format_all_queue_sizes(all_queues))
        time.sleep(5)


def portrait_norm_consumer(task_queue, processing_set, set_lock, image_edit_runner, worker_id=0):
    tag = f"[人物标准化 消费者-{worker_id} ({image_edit_runner.device})]"
    print(f"{tag} 已启动...")
    while True:
        try:
            mb, clip_base, image_name = task_queue.get(timeout=3)
        except queue.Empty:
            continue

        portraits_dir = os.path.join(OUTPUT_DIR, mb, "asd", clip_base, "portraits_yolo")
        in_path = os.path.join(portraits_dir, image_name)
        out_path = os.path.join(portraits_dir, f"{os.path.splitext(image_name)[0]}_norm.jpg")
        item_id = f"{mb}/{clip_base}/{image_name}"

        try:
            ok = image_edit_runner.process(
                input_image_path=in_path,
                output_image_path=out_path,
                face_visible=IMAGE_EDIT_DEFAULT_FACE_VISIBLE,
                shot_type=IMAGE_EDIT_DEFAULT_SHOT_TYPE,
            )
            if not ok and not os.path.exists(out_path):
                open(out_path, "a").close()
            print(f"{tag} {mb} -> {clip_base}/{image_name} {'完成' if ok else '失败'}")
        except Exception as e:
            if not os.path.exists(out_path):
                open(out_path, "a").close()
            print(f"{tag} {mb} -> {clip_base}/{image_name} 异常/失败，异常信息: {repr(e)}")
        finally:
            with set_lock:
                processing_set.discard(item_id)
            task_queue.task_done()


# ==========================================
# 主程序入口
# ==========================================
def main():
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token: print("⚠️ 警告: 找不到 HuggingFace Token！")
        
    print("正在初始化所有模型 (显存消耗极大，建议监控 nvidia-smi)...")
    # transnet = TransNetV2Pipeline("Models/TransNetV2", "cuda:2")
    # person_pipe = PersonDetectionPipeline(sample_interval_sec=1.0, min_person_frames=1, confidence_threshold=0.5)
    # ocr_pipe = EasyOCRCreditDetectorPipeline(sample_interval_sec=1.0, min_text_boxes=1)
    # whisper_pipe = MovieTranscriber(hf_token=hf_token, device="cuda", compute_type="float16")
    # ClearVoice 固定使用 cuda:0，避免多卡环境下张量设备不一致 (weight on cuda:1 vs input on cuda:0)
    # cv_pipe = ClearVoicePipeline(task="speech_enhancement", model_name="MossFormer2_SE_48K", device="cuda:0")
    print(f"正在初始化 {len(IMAGE_EDIT_DEVICES)} 个 Qwen Image Edit Runner: {IMAGE_EDIT_DEVICES}")
    image_edit_runners = [
        QwenImageEditRunner(
            device=device,
            default_face_visible=IMAGE_EDIT_DEFAULT_FACE_VISIBLE,
            default_shot_type=IMAGE_EDIT_DEFAULT_SHOT_TYPE,
        )
        for device in IMAGE_EDIT_DEVICES
    ]
    for image_edit_runner in image_edit_runners:
        image_edit_runner.ensure_loaded()

    # print(f"正在初始化 {len(ASD_DEVICES)} 个 ASD Pipeline: {ASD_DEVICES}")
    # asd_pipes = [LightASDPipeline(device=dev) for dev in ASD_DEVICES]

    print("🚀 Qwen 人物标准化流水线启动！")

    queues = {k: queue.Queue() for k in ["transnet", "clips", "whisper", "asd", "clearvoice", "portrait_norm"]}
    locks = {k: threading.Lock() for k in ["transnet", "clips", "whisper", "asd", "clearvoice", "portrait_norm"]}
    sets = {k: set() for k in ["transnet", "clips", "whisper", "asd", "clearvoice", "portrait_norm"]}

    # 非 Qwen 阶段暂时停用，仅保留人物标准化链路排查显存问题。
    # other_stages = [
    #     ("transnet", transnet_producer, transnet_consumer, transnet),
    #     ("clips", clips_check_producer, clips_check_consumer, (ocr_pipe, person_pipe)),
    #     ("whisper", whisperx_producer, whisperx_consumer, whisper_pipe),
    #     ("clearvoice", clearvoice_producer, clearvoice_consumer, cv_pipe),
    # ]
    # for stage_name, prod, cons, pipe in other_stages:
    #     threading.Thread(target=prod, args=(queues[stage_name], sets[stage_name], locks[stage_name], queues), daemon=True).start()
    #     if stage_name == "clips":
    #         threading.Thread(target=cons, args=(queues[stage_name], sets[stage_name], locks[stage_name], pipe[0], pipe[1]), daemon=True).start()
    #     else:
    #         threading.Thread(target=cons, args=(queues[stage_name], sets[stage_name], locks[stage_name], pipe), daemon=True).start()

    # 人物标准化阶段：1 生产者 + N 消费者（每张卡一个）
    threading.Thread(
        target=portrait_norm_producer,
        args=(queues["portrait_norm"], sets["portrait_norm"], locks["portrait_norm"], queues),
        daemon=True,
    ).start()
    for i, image_edit_runner in enumerate(image_edit_runners):
        threading.Thread(
            target=portrait_norm_consumer,
            args=(queues["portrait_norm"], sets["portrait_norm"], locks["portrait_norm"], image_edit_runner),
            kwargs={"worker_id": i},
            daemon=True,
        ).start()

    # ASD 阶段暂时停用，仅保留 Qwen 阶段单独测试。
    # threading.Thread(target=asd_producer, args=(queues["asd"], sets["asd"], locks["asd"], queues), daemon=True).start()
    # for i, asd_pipe in enumerate(asd_pipes):
    #     threading.Thread(
    #         target=asd_consumer,
    #         args=(queues["asd"], sets["asd"], locks["asd"], asd_pipe),
    #         kwargs={"worker_id": i},
    #         daemon=True,
    #     ).start()

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到中断信号，程序正在安全退出...")

if __name__ == "__main__":
    main()
