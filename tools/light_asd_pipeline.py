import os
import sys
import cv2
import time
import math
import torch
import numpy as np
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
import python_speech_features
import shutil
import warnings
import logging
from tqdm import tqdm

# 忽略全局的各种弃用警告
warnings.filterwarnings("ignore")
# 关闭 scenedetect 的底层日志
logging.getLogger('scenedetect').setLevel(logging.ERROR)


# 处理状态常量：用于 process_status.json，明确标记处理结果，便于跳过逻辑解耦
PROCESS_STATUS_COMPLETED = "COMPLETED"
PROCESS_STATUS_NO_TRACKABLE_FACE = "NO_TRACKABLE_FACE"
PROCESS_STATUS_NO_VALID_TRACK = "NO_VALID_TRACK"
PROCESS_STATUS_NO_VALID_SPEAKER = "NO_VALID_SPEAKER"
PROCESS_STATUS_FAILED = "FAILED"
PROCESS_STATUS_SKIP_VALUES = frozenset({
    PROCESS_STATUS_COMPLETED,
    PROCESS_STATUS_NO_TRACKABLE_FACE,
    PROCESS_STATUS_NO_VALID_TRACK,
    PROCESS_STATUS_NO_VALID_SPEAKER,
})


def _write_process_status(ws_dir, status, message=None, extra=None):
    """在 results 目录下写入 process_status.json，用于显式标记处理状态。"""
    results_dir = os.path.join(ws_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    status_file = os.path.join(results_dir, "process_status.json")
    payload = {"status": status}
    if message:
        payload["message"] = message
    if extra:
        payload.update(extra)
    try:
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.getLogger(__name__).warning("写入 process_status.json 失败: %s", e)


def _read_process_status(ws_dir):
    """读取 process_status.json，若不存在或解析失败返回 None。"""
    status_file = os.path.join(ws_dir, "results", "process_status.json")
    if not os.path.isfile(status_file):
        return None
    try:
        with open(status_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def to_jsonable(obj):
    """
    递归将 numpy / tuple 等对象转成可 JSON 序列化结构。
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return obj


class HiddenOutputs:
    """去除全局 stdout 重定向，防止多线程输出丢失或静默崩溃"""

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# 动态将 Light-ASD 目录加入系统路径，确保能引入它的包（基于当前文件所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIGHT_ASD_DIR = os.path.join(BASE_DIR, "Light-ASD")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.workspace import ASDWorkspace
from video_io.video_utils import (
    extract_video_to_workspace,
    extract_audio_segment,
    mux_video_audio,
    transcode_with_fallback as video_transcode_with_fallback,
)
from utils.subtitle_aligner import (
    format_time,
    get_speaking_segments,
    pick_target_track_by_asd,
    align_subtitles_with_speakers,
)
if LIGHT_ASD_DIR not in sys.path:
    sys.path.append(LIGHT_ASD_DIR)

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from ASD import ASD

class LightASDPipeline:
    def __init__(
        self,
        pretrain_model_path=None,
        device='cuda',
        face_conf_th=0.8,
        min_track_frames=5,
        min_shot_sec=0.5,
        precheck_face_conf_th=0.6,
        precheck_sample_interval_sec=0.5,
        precheck_min_face_hits=1,
    ):
        self.device = device
        # 默认使用项目内 tools/Light-ASD/weight 下的权重
        if pretrain_model_path is None:
            self.pretrain_model_path = os.path.join(LIGHT_ASD_DIR, "weight", "pretrain_AVA_CVPR.model")
        else:
            self.pretrain_model_path = pretrain_model_path
        self.n_threads = 2  # 多视频并行时降低 FFmpeg 线程，避免 CPU 过载
        self.face_conf_th = float(face_conf_th)
        self.min_track_frames = int(min_track_frames)
        self.min_shot_sec = float(min_shot_sec)
        self.precheck_face_conf_th = float(precheck_face_conf_th)
        self.precheck_sample_interval_sec = float(precheck_sample_interval_sec)
        self.precheck_min_face_hits = int(precheck_min_face_hits)
        self.face_detector = None
        self.asd_model = None
        self.last_track_mp4_paths = []

    def _load_models(self):
        """懒加载模型，避免初始化 Pipeline 时就占满显存"""
        if self.face_detector is None:
            with HiddenOutputs():  # 屏蔽 S3FD 内部可能的输出
                self.face_detector = S3FD(device=self.device)
        
        if self.asd_model is None:
            with HiddenOutputs():  # 屏蔽 Light-ASD 加载时的 print（如 "Model para number"）
                self.asd_model = ASD()
                self.asd_model.loadParameters(self.pretrain_model_path)
            self.asd_model.to(self.device)
            self.asd_model.eval()

    def _load_yolo(self):
        """懒加载 YOLO 模型，避免不必要时占用显存"""
        if not hasattr(self, 'yolo_model') or self.yolo_model is None:
            try:
                from ultralytics import YOLO
                import logging
                # 关闭 YOLO 繁琐的推理日志
                logging.getLogger("ultralytics").setLevel(logging.ERROR)
                # 加载最轻量级的 YOLOv8n，速度极快，足够找全身框
                self.yolo_model = YOLO('yolov8n.pt') 
            except ImportError:
                print("[ERROR] 缺少 ultralytics 库，请运行: pip install ultralytics")
                sys.exit(1)

    def bb_intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    def track_shot(self, sceneFaces):
        """人脸追踪逻辑"""
        iouThres  = 0.5 
        minTrack  = self.min_track_frames
        numFailedDet = 10
        minFaceSize = 1
        tracks = []
        
        while True:
            track = []
            for frameFaces in sceneFaces:
                for face in frameFaces:
                    if track == []:
                        track.append(face)
                        frameFaces.remove(face)
                    elif face['frame'] - track[-1]['frame'] <= numFailedDet:
                        iou = self.bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                        if iou > iouThres:
                            track.append(face)
                            frameFaces.remove(face)
                            continue
                    else:
                        break
            if track == []:
                break
            elif len(track) >= minTrack:
                frameNum = np.array([f['frame'] for f in track])
                bboxes   = np.array([np.array(f['bbox']) for f in track])
                frameI   = np.arange(frameNum[0], frameNum[-1]+1)
                bboxesI  = []
                for ij in range(0,4):
                    interpfn = interp1d(frameNum, bboxes[:,ij])
                    bboxesI.append(interpfn(frameI))
                bboxesI = np.stack(bboxesI, axis=1)
                if max(np.mean(bboxesI[:,2]-bboxesI[:,0]), np.mean(bboxesI[:,3]-bboxesI[:,1])) > minFaceSize:
                    tracks.append({'frame':frameI, 'bbox':bboxesI})
        return tracks

    def _has_trackable_face_fast(self, video_file_path):
        """
        快速预检：抽样若干帧，只要检测到足够人脸就进入完整 ASD。
        """
        self._load_models()
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-6:
            fps = 25.0
        step = max(1, int(round(fps * self.precheck_sample_interval_sec)))

        frame_idx = 0
        face_hits = 0
        while True:
            ok, frame_img = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            image_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            bboxes = self.face_detector.detect_faces(
                image_rgb,
                conf_th=self.precheck_face_conf_th,
                scales=[0.25],
            )
            if bboxes is not None and len(bboxes) > 0:
                face_hits += 1
                if face_hits >= self.precheck_min_face_hits:
                    cap.release()
                    return True
            frame_idx += 1

        cap.release()
        return False

    def crop_video(self, track, pyframes_path, audio_path, cropFile):
        """切割视频与音频"""
        flist = [os.path.join(pyframes_path, f) for f in os.listdir(pyframes_path) if f.endswith('.jpg')]
        flist.sort()
        vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))
        dets = {'x':[], 'y':[], 's':[]}
        
        for det in track['bbox']:
            dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
            dets['y'].append((det[1]+det[3])/2) 
            dets['x'].append((det[0]+det[2])/2) 
            
        dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  
        dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
        dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
        
        cropScale = 0.40
        for fidx, frame in enumerate(track['frame']):
            bs  = dets['s'][fidx]   
            bsi = int(bs * (1 + 2 * cropScale))  
            image = cv2.imread(flist[frame])
            frame_img = np.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
            my  = dets['y'][fidx] + bsi  
            mx  = dets['x'][fidx] + bsi  
            face = frame_img[int(my-bs):int(my+bs*(1+2*cropScale)), int(mx-bs*(1+cropScale)):int(mx+bs*(1+cropScale))]
            vOut.write(cv2.resize(face, (224, 224)))
            
        vOut.release()
        audioTmp = cropFile + '.wav'
        audioStart = (track['frame'][0]) / 25
        audioEnd = (track['frame'][-1] + 1) / 25

        extract_audio_segment(
            audio_path,
            audioTmp,
            audioStart,
            audioEnd,
            n_threads=self.n_threads,
        )
        mux_video_audio(
            cropFile + 't.avi',
            audioTmp,
            cropFile + '.avi',
            n_threads=self.n_threads,
        )
        os.remove(cropFile + 't.avi')
        
        return {'track':track, 'proc_track':dets}

    def evaluate_network(self, files):
        """推理打分"""
        allScores = []
        durationSet = {1,1,1,2,2,2,3,3,4,5,6} 
        for file in files:
            fileName = os.path.splitext(file)[0] 
            _, audio = wavfile.read(fileName + '.wav')
            audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
            
            video = cv2.VideoCapture(fileName + '.avi')
            videoFeature = []
            while video.isOpened():
                ret, frames = video.read()
                if ret:
                    face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (224,224))
                    face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                    videoFeature.append(face)
                else:
                    break
            video.release()
            
            videoFeature = np.array(videoFeature)
            length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
            audioFeature = audioFeature[:int(round(length * 100)),:]
            videoFeature = videoFeature[:int(round(length * 25)),:,:]
            allScore = [] 
            
            for duration in durationSet:
                batchSize = int(math.ceil(length / duration))
                scores = []
                with torch.no_grad():
                    for i in range(batchSize):
                        inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).to(self.device)
                        inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).to(self.device)
                        embedA = self.asd_model.model.forward_audio_frontend(inputA)
                        embedV = self.asd_model.model.forward_visual_frontend(inputV)   
                        out = self.asd_model.model.forward_audio_visual_backend(embedA, embedV)
                        score = self.asd_model.lossAV.forward(out, labels = None)
                        
                        if torch.is_tensor(score):
                            scores.extend(score.tolist())
                        else:
                            scores.extend(score)
                            
                allScore.append(scores)
            allScore = np.round((np.mean(np.array(allScore), axis=0)), 1).astype(float)
            allScores.append(allScore.tolist()) 
        return allScores

    def export_track_mp4_with_bbox(
        self,
        video_file_path,
        audio_file_path,
        track_info,
        track_scores,
        output_mp4_path,
        fps=25,
        threshold=-0.5,
    ):
        """将单个 track 导出为原视频视角的 mp4，并绘制该人物检测框。"""
        track = track_info.get("track", {})
        frame_seq = track.get("frame", [])
        bbox_seq = track.get("bbox", [])
        if len(frame_seq) == 0 or len(bbox_seq) == 0:
            return False

        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 1e-6:
            video_fps = fps

        raw_mp4_path = output_mp4_path + ".raw.mp4"
        writer = cv2.VideoWriter(
            raw_mp4_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (width, height),
        )

        start_frame = int(frame_seq[0])
        end_frame = int(frame_seq[-1])

        for frame_num in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, frame_img = cap.read()
            if not ok:
                continue

            rel_idx = frame_num - start_frame
            if rel_idx < len(bbox_seq):
                x1, y1, x2, y2 = bbox_seq[rel_idx]
                x1 = int(max(0, min(width - 1, x1)))
                y1 = int(max(0, min(height - 1, y1)))
                x2 = int(max(0, min(width - 1, x2)))
                y2 = int(max(0, min(height - 1, y2)))

                score = None
                if rel_idx < len(track_scores):
                    score = float(track_scores[rel_idx])
                is_speaking = score is not None and score >= threshold

                color = (0, 255, 0) if is_speaking else (0, 165, 255)  # BGR
                status = "speaking" if is_speaking else "silent"
                score_text = f"{score:.2f}" if score is not None else "NA"
                label = f"Track | {status} | score={score_text}"

                cv2.rectangle(frame_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame_img,
                    label,
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame_img)

        writer.release()
        cap.release()

        start_sec = float(start_frame) / float(video_fps)
        end_sec = float(end_frame + 1) / float(video_fps)

        ok, used_vcodec, err = self._transcode_with_fallback(
            raw_mp4_path=raw_mp4_path,
            audio_file_path=audio_file_path,
            output_mp4_path=output_mp4_path,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        if not ok:
            if os.path.exists(output_mp4_path):
                os.remove(output_mp4_path)
            if os.path.exists(raw_mp4_path):
                os.remove(raw_mp4_path)
            print(f"[WARN] ffmpeg 导出 {output_mp4_path} 失败: {err}")
            return False
        print(f"[INFO] 导出 {output_mp4_path} 使用视频编码器: {used_vcodec}")

        if os.path.exists(raw_mp4_path):
            os.remove(raw_mp4_path)
        return True

    def export_all_tracks_mp4_with_bbox(
        self,
        video_file_path,
        audio_file_path,
        vid_tracks,
        scores,
        output_mp4_path,
        fps=25,
        threshold=-0.5,
    ):
        """将所有 track 叠加到同一个全长视频中导出。"""
        if not vid_tracks or not scores:
            return False

        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 1e-6:
            video_fps = fps

        raw_mp4_path = output_mp4_path + ".raw.mp4"
        writer = cv2.VideoWriter(
            raw_mp4_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (width, height),
        )

        frame_to_items = {}
        for track_idx, (track_info, track_scores) in enumerate(zip(vid_tracks, scores)):
            track = track_info.get("track", {})
            frame_seq = track.get("frame", [])
            bbox_seq = track.get("bbox", [])
            valid_len = min(len(frame_seq), len(bbox_seq))
            for rel_idx in range(valid_len):
                frame_num = int(frame_seq[rel_idx])
                bbox = bbox_seq[rel_idx]
                score = float(track_scores[rel_idx]) if rel_idx < len(track_scores) else None
                is_speaking = score is not None and score >= threshold
                frame_to_items.setdefault(frame_num, []).append(
                    {
                        "track_idx": track_idx,
                        "bbox": bbox,
                        "score": score,
                        "is_speaking": is_speaking,
                    }
                )

        frame_idx = 0
        while True:
            ok, frame_img = cap.read()
            if not ok:
                break

            for item in frame_to_items.get(frame_idx, []):
                x1, y1, x2, y2 = item["bbox"]
                x1 = int(max(0, min(width - 1, x1)))
                y1 = int(max(0, min(height - 1, y1)))
                x2 = int(max(0, min(width - 1, x2)))
                y2 = int(max(0, min(height - 1, y2)))

                color = (0, 255, 0) if item["is_speaking"] else (0, 165, 255)  # BGR
                status = "spk" if item["is_speaking"] else "sil"
                score_text = f"{item['score']:.2f}" if item["score"] is not None else "NA"
                label = f"T{item['track_idx']:02d} {status} {score_text}"

                cv2.rectangle(frame_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame_img,
                    label,
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame_img)
            frame_idx += 1

        writer.release()
        cap.release()

        ok, used_vcodec, err = self._transcode_with_fallback(
            raw_mp4_path=raw_mp4_path,
            audio_file_path=audio_file_path,
            output_mp4_path=output_mp4_path,
            start_sec=None,
            end_sec=None,
        )
        if not ok:
            if os.path.exists(output_mp4_path):
                os.remove(output_mp4_path)
            if os.path.exists(raw_mp4_path):
                os.remove(raw_mp4_path)
            print(f"[WARN] ffmpeg 导出全轨迹视频 {output_mp4_path} 失败: {err}")
            return False
        print(f"[INFO] 导出全轨迹视频 {output_mp4_path} 使用视频编码器: {used_vcodec}")

        if os.path.exists(raw_mp4_path):
            os.remove(raw_mp4_path)
        return True

    def render_aligned_subtitles_on_video(
        self,
        input_video_path,
        audio_file_path,
        aligned_subtitles,
        output_mp4_path,
        fps=25,
    ):
        """将“已匹配说话人ID”的台词直接绘制到视频画面上。"""
        if not os.path.isfile(input_video_path):
            print(f"[WARN] 输入视频不存在: {input_video_path}")
            return False
        if not aligned_subtitles:
            print("[WARN] 无台词可绘制，跳过字幕视频导出。")
            return False

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"[WARN] 无法打开输入视频: {input_video_path}")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 1e-6:
            video_fps = fps

        raw_mp4_path = output_mp4_path + ".raw.mp4"
        writer = cv2.VideoWriter(
            raw_mp4_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (width, height),
        )

        subtitles = sorted(aligned_subtitles, key=lambda x: (x["start"], x["end"]))

        def _fit_text(text, max_len=72):
            if len(text) <= max_len:
                return text
            return text[: max_len - 3] + "..."

        frame_idx = 0
        while True:
            ok, frame_img = cap.read()
            if not ok:
                break

            current_time = frame_idx / float(video_fps)
            active_texts = []
            for sub in subtitles:
                if sub["start"] <= current_time <= sub["end"]:
                    active_texts.append(_fit_text(sub.get("display_text", sub.get("text", ""))))

            if active_texts:
                line_h = 34
                box_h = 16 + line_h * len(active_texts)
                y1 = max(0, height - box_h - 18)
                y2 = height - 10
                cv2.rectangle(frame_img, (16, y1), (width - 16, y2), (0, 0, 0), -1)

                for i, txt in enumerate(active_texts):
                    y = y1 + 26 + i * line_h
                    cv2.putText(
                        frame_img,
                        txt,
                        (28, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            writer.write(frame_img)
            frame_idx += 1

        writer.release()
        cap.release()

        ok, used_vcodec, err = self._transcode_with_fallback(
            raw_mp4_path=raw_mp4_path,
            audio_file_path=audio_file_path,
            output_mp4_path=output_mp4_path,
            start_sec=None,
            end_sec=None,
        )
        if not ok:
            if os.path.exists(output_mp4_path):
                os.remove(output_mp4_path)
            if os.path.exists(raw_mp4_path):
                os.remove(raw_mp4_path)
            print(f"[WARN] 带台词视频导出失败 {output_mp4_path}: {err}")
            return False

        if os.path.exists(raw_mp4_path):
            os.remove(raw_mp4_path)
        print(f"[INFO] 已导出带台词视频: {output_mp4_path} | 编码器: {used_vcodec}")
        return True

    def _transcode_with_fallback(
        self,
        raw_mp4_path,
        audio_file_path,
        output_mp4_path,
        start_sec,
        end_sec,
    ):
        """使用 ffmpeg 转码并封装音频，自动回退视频编码器。"""
        return video_transcode_with_fallback(
            raw_mp4_path,
            audio_file_path,
            output_mp4_path,
            start_sec=start_sec,
            end_sec=end_sec,
        )

    def _to_builtin(self, obj):
        if isinstance(obj, dict):
            return {k: self._to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_builtin(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    def save_tracking_and_asd_results(self, ws_dir, vid_tracks, scores, threshold):
        """保存 tracking 与 ASD 结果到 workspace/results 目录。"""
        results_dir = os.path.join(ws_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        tracking_path = os.path.join(results_dir, "tracking_results.json")
        asd_path = os.path.join(results_dir, "asd_scores.json")
        merged_path = os.path.join(results_dir, "tracking_asd_merged.json")

        tracking_payload = {
            "num_tracks": len(vid_tracks),
            "tracks": self._to_builtin(vid_tracks),
        }
        asd_payload = {
            "num_tracks_with_scores": len(scores),
            "threshold": float(threshold),
            "scores": self._to_builtin(scores),
        }
        merged_payload = {
            "num_tracks": len(vid_tracks),
            "threshold": float(threshold),
            "tracks": self._to_builtin(vid_tracks),
            "scores": self._to_builtin(scores),
        }

        with open(tracking_path, "w", encoding="utf-8") as f:
            json.dump(tracking_payload, f, ensure_ascii=False, indent=2)
        with open(asd_path, "w", encoding="utf-8") as f:
            json.dump(asd_payload, f, ensure_ascii=False, indent=2)
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(merged_payload, f, ensure_ascii=False, indent=2)

        print(f"[INFO] 已保存 tracking 结果: {tracking_path}")
        print(f"[INFO] 已保存 ASD 结果: {asd_path}")
        print(f"[INFO] 已保存合并结果: {merged_path}")
        return {
            "tracking_path": tracking_path,
            "asd_path": asd_path,
            "merged_path": merged_path,
        }

    def process_video(self, video_path, workspace_base="./asd_workspace", threshold=-0.5):
        """主处理流水线"""
        self._load_models()
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        ws = ASDWorkspace(workspace_base, video_name)
        ws.init_dirs()
        ws_dir = ws.root

        # 1. 音视频分离提取
        extract_video_to_workspace(
            video_path,
            ws.video_avi,
            ws.audio_wav,
            ws.frames_pattern,
            n_threads=self.n_threads,
        )

        # 1.5 快速人脸预检：无人脸可追踪则跳过，避免无效 ASD 开销
        if not self._has_trackable_face_fast(ws.video_avi):
            print(f"[INFO] 预检未发现可追踪人脸，跳过 ASD: {video_path}")
            try:
                tracking_json_path = ws.tracking_json
                asd_json_path = ws.asd_results_json
                with open(tracking_json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "video_path": video_path,
                            "workspace": ws_dir,
                            "num_all_tracks": 0,
                            "num_vid_tracks": 0,
                            "all_tracks": [],
                            "vid_tracks": [],
                            "skip_reason": "precheck_no_trackable_face",
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                with open(asd_json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "video_path": video_path,
                            "workspace": ws_dir,
                            "threshold": threshold,
                            "num_scores": 0,
                            "scores": [],
                            "speaking_segments": [],
                            "skip_reason": "precheck_no_trackable_face",
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                _write_process_status(
                    ws_dir,
                    PROCESS_STATUS_NO_TRACKABLE_FACE,
                    message="预检未发现可追踪人脸",
                )
            except Exception as e:
                print(f"[WARN] 预检跳过时写结果失败: {e}")
            ws.cleanup_frames()
            return [], []

        # 2. 场景检测
        with HiddenOutputs():
            videoManager = VideoManager([ws.video_avi])
            sceneManager = SceneManager(StatsManager())
            sceneManager.add_detector(ContentDetector())
            videoManager.set_downscale_factor()
            videoManager.start()
            sceneManager.detect_scenes(frame_source=videoManager)
            sceneList = sceneManager.get_scene_list(videoManager.get_base_timecode())
            if not sceneList:
                sceneList = [(videoManager.get_base_timecode(), videoManager.get_current_timecode())]

        # 3. 抽帧人脸检测
        flist = [os.path.join(ws.pyframes, f) for f in os.listdir(ws.pyframes) if f.endswith('.jpg')]
        flist.sort()
        faces = []
        for fidx, fname in enumerate(flist):
            image = cv2.imread(fname)
            imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes = self.face_detector.detect_faces(imageNumpy, conf_th=self.face_conf_th, scales=[0.25])
            faces.append([{'frame': fidx, 'bbox': bbox[:-1].tolist(), 'conf': bbox[-1]} for bbox in bboxes])

        # 4. 人脸追踪
        allTracks, vidTracks = [], []
        min_shot_frames = max(1, int(round(25.0 * self.min_shot_sec)))
        for shot in sceneList:
            if shot[1].frame_num - shot[0].frame_num >= min_shot_frames:
                allTracks.extend(self.track_shot(faces[shot[0].frame_num:shot[1].frame_num]))
                
        # 5. 切片剪辑
        for ii, track in enumerate(allTracks):
            cropFile = os.path.join(ws.pycrop, f'{ii:05d}')
            vidTracks.append(self.crop_video(track, ws.pyframes, ws.audio_wav, cropFile))

        # 6. ASD 推理
        files = [os.path.join(ws.pycrop, f) for f in os.listdir(ws.pycrop) if f.endswith('.avi')]
        files.sort()
        scores = self.evaluate_network(files)

        # 6.5 保存 tracking / asd 结果
        try:
            tracking_json_path = ws.tracking_json
            asd_json_path = ws.asd_results_json
            speaking_segments = get_speaking_segments(
                vidTracks,
                scores,
                fps=25,
                threshold=threshold,
                min_duration=0.0,
            )

            tracking_payload = {
                "video_path": video_path,
                "workspace": ws_dir,
                "num_all_tracks": len(allTracks),
                "num_vid_tracks": len(vidTracks),
                "all_tracks": to_jsonable(allTracks),
                "vid_tracks": to_jsonable(vidTracks),
            }
            asd_payload = {
                "video_path": video_path,
                "workspace": ws_dir,
                "threshold": threshold,
                "num_scores": len(scores),
                "scores": to_jsonable(scores),
                "speaking_segments": to_jsonable(speaking_segments),
            }

            with open(tracking_json_path, "w", encoding="utf-8") as f:
                json.dump(tracking_payload, f, ensure_ascii=False, indent=2)
            with open(asd_json_path, "w", encoding="utf-8") as f:
                json.dump(asd_payload, f, ensure_ascii=False, indent=2)

            print(f"[INFO] tracking 结果已保存: {tracking_json_path}")
            print(f"[INFO] asd 结果已保存: {asd_json_path}")
        except Exception as e:
            print(f"[WARN] 保存 tracking/asd 结果失败: {e}")

        # 7. 导出“全 track 同屏”的全长视频
        self.last_track_mp4_paths = []
        ok = self.export_all_tracks_mp4_with_bbox(
            video_file_path=ws.video_avi,
            audio_file_path=ws.audio_wav,
            vid_tracks=vidTracks,
            scores=scores,
            output_mp4_path=ws.all_tracks_mp4,
            fps=25,
            threshold=threshold,
        )
        if ok:
            self.last_track_mp4_paths.append(ws.all_tracks_mp4)
            print(f"已导出全轨迹同屏视频(全长): {ws.all_tracks_mp4}")
        else:
            print("未导出全轨迹同屏视频（可能没有有效轨迹）。")

        # 8. 保存 tracking 与 ASD 结果
        self.save_tracking_and_asd_results(
            ws_dir=ws_dir,
            vid_tracks=vidTracks,
            scores=scores,
            threshold=threshold,
        )

        # 清理帧截图
        ws.cleanup_frames()

        return vidTracks, scores

    def export_speaker_portraits_yolo(self, video_file_path, vid_tracks, output_dir):
        """
        利用 YOLO 提取每个说话人的全身/大半身画面。
        思路：在人脸【最小】的那一帧（即全景，身体信息最全），用 YOLO 检测人体，找到包含该人脸的人体框并裁剪。
        """
        self._load_yolo()
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            print(f"[WARN] 无法打开视频以提取头像: {video_file_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for track_idx, track_info in enumerate(vid_tracks):
            track = track_info.get("track", {})
            frame_seq = track.get("frame", [])
            bbox_seq = track.get("bbox", [])

            if len(frame_seq) == 0 or len(bbox_seq) == 0:
                continue

            # 1. 找到该轨迹中人脸框面积【最小】的一帧（离镜头最远，身体保留最完整）
            min_area = float('inf')
            best_idx = 0
            for i, bbox in enumerate(bbox_seq):
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    min_area = area
                    best_idx = i

            best_frame_num = int(frame_seq[best_idx])
            face_bbox = bbox_seq[best_idx]
            fx1, fy1, fx2, fy2 = face_bbox

            cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_num)
            ok, frame_img = cap.read()
            if not ok:
                continue

            # 2. 使用 YOLO 在该帧中检测人体 (class=0 表示 person)
            results = self.yolo_model(frame_img, classes=[0], verbose=False)
            
            best_person_bbox = None
            max_intersection = 0

            # 3. 匹配包含该人脸的人体框
            for r in results:
                for box in r.boxes:
                    px1, py1, px2, py2 = box.xyxy[0].cpu().numpy()
                    
                    # 计算人脸和人体的交集面积
                    ix1 = max(fx1, px1)
                    iy1 = max(fy1, py1)
                    ix2 = min(fx2, px2)
                    iy2 = min(fy2, py2)
                    
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        # 找到交集面积最大的人体框，即为该说话人对应的身体
                        if inter_area > max_intersection:
                            max_intersection = inter_area
                            best_person_bbox = [int(px1), int(py1), int(px2), int(py2)]
            
            # 容错处理：如果 YOLO 没检测到人，使用扩大面部框进行兜底
            if best_person_bbox is None:
                print(f"[WARN] Track_{track_idx:05d} 未被 YOLO 识别到人体，使用扩大面部框容错。")
                px1 = max(0, int(fx1 - (fx2-fx1)*1.5))
                py1 = max(0, int(fy1 - (fy2-fy1)*0.5))
                px2 = min(width, int(fx2 + (fx2-fx1)*1.5))
                py2 = min(height, int(fy2 + (fy2-fy1)*4.0))
            else:
                px1, py1, px2, py2 = best_person_bbox
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(width, px2), min(height, py2)

            person_crop = frame_img[py1:py2, px1:px2]

            # 4. 保存图片，命名与说话人一致 (例如 Track_00000.jpg)
            speaker_name = f"Track_{track_idx:05d}"
            save_path = os.path.join(output_dir, f"{speaker_name}.jpg")
            
            if person_crop.size > 0:
                cv2.imwrite(save_path, person_crop)
                print(f"[INFO] 已使用 YOLO 保存 {speaker_name} 的整个人像: {save_path}")

        cap.release()


def _load_existing_clips_check(jsonl_path):
    """加载已有的 clips_check.jsonl，返回 {filename: record}"""
    if not os.path.isfile(jsonl_path):
        return {}
    cached = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                fn = record.get("file")
                if fn:
                    cached[fn] = record
            except json.JSONDecodeError:
                continue
    return cached


def _has_real_dialogue(json_path):
    """
    检查 whisperx JSON 中是否有真实台词（非空、非纯标点/空白）。
    返回 True 表示有台词，False 表示无台词或解析失败。
    """
    if not os.path.isfile(json_path):
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False

    texts = []
    segments = data.get("segments")
    if isinstance(segments, list):
        for seg in segments:
            if isinstance(seg, dict):
                t = seg.get("text") or seg.get("word")
                if t and isinstance(t, str):
                    texts.append(t.strip())
    word_segments = data.get("word_segments")
    if isinstance(word_segments, list) and not texts:
        for w in word_segments:
            if isinstance(w, dict):
                t = w.get("word") or w.get("text")
                if t and isinstance(t, str):
                    texts.append(t.strip())

    combined = "".join(texts).strip()
    return bool(combined) and any(c.isalnum() for c in combined)


def _duration_sec_from_shot_basename(base_name, fps=24):
    """
    从镜头文件名解析时长（秒）。格式如 scene_001_000000_000097 表示 0～97 帧。
    返回 (end - start + 1) / fps，解析失败时返回 None（表示不过滤）。
    """
    parts = base_name.split("_")
    if len(parts) < 2:
        return None
    try:
        start_frame = int(parts[-2])
        end_frame = int(parts[-1])
        frame_count = end_frame - start_frame + 1
        return float(frame_count) / float(fps)
    except (ValueError, IndexError):
        return None


def _collect_asd_targets(movie_output_dir, min_duration_sec=2.0, shot_fps=24):
    """
    收集可执行 ASD 的镜头（与 app.py 逻辑一致）：
    返回 [(shot_path, base_name, srt_path, whisper_json_path), ...]
    仅保留满足以下条件的片段：
    1) clips_check 里 has_person=True 且 has_text=False（有人物且无文字）
    2) 同时存在 whisperx srt + whisperx json
    3) 视频时长 >= min_duration_sec（默认 2 秒，低于此值不处理；从文件名 scene_XXX_AAAAAA_BBBBBB 解析帧数，按 shot_fps 换算）
    4) whisperx json 中必须有真实台词（非空、非纯标点）
    """
    shots_dir = os.path.join(movie_output_dir, "shots")
    subtitles_dir = os.path.join(movie_output_dir, "subtitles")
    clips_check_jsonl = os.path.join(movie_output_dir, "clips_check.jsonl")
    if not os.path.isdir(shots_dir) or not os.path.isdir(subtitles_dir) or not os.path.isfile(clips_check_jsonl):
        return []

    clip_records = _load_existing_clips_check(clips_check_jsonl)
    targets = []
    for filename in tqdm(sorted(f for f in os.listdir(shots_dir) if f.lower().endswith(".mp4"))):
        rec = clip_records.get(filename, {})
        if rec.get("has_person") is not True:
            continue
        if rec.get("has_text") is not False:
            continue

        base = os.path.splitext(filename)[0]
        shot_path = os.path.join(shots_dir, filename)
        srt_path = os.path.join(subtitles_dir, f"{base}_whisperx.srt")
        json_path = os.path.join(subtitles_dir, f"{base}_whisperx.json")
        if not os.path.isfile(srt_path) or not os.path.isfile(json_path):
            continue

        duration = _duration_sec_from_shot_basename(base, fps=shot_fps)
        if duration is not None and duration < min_duration_sec:
            continue

        if not _has_real_dialogue(json_path):
            continue

        targets.append((shot_path, base, srt_path, json_path))
    return targets


def _process_one_target(
    pipeline,
    pipeline_lock,
    task,
    threshold,
    skip_done,
    fps,
):
    """
    处理单个 ASD 目标。在多卡模式下由工作线程调用，需持有 pipeline_lock 以保证同一 GPU 上的模型不被并发使用。
    """
    movie_dir, movie_name, (shot_path, base, srt_path, json_path) = task
    asd_root = os.path.join(movie_dir, "asd")
    ws_dir = os.path.join(asd_root, base)
    result_dir = os.path.join(ws_dir, "results")
    pytracks_dir = os.path.join(ws_dir, "pytracks")
    portraits_dir = os.path.join(ws_dir, "portraits_yolo")
    subtitles_dir = os.path.join(movie_dir, "subtitles")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pytracks_dir, exist_ok=True)

    out_mp4 = os.path.join(pytracks_dir, "all_tracks_with_dialog.mp4")
    mapping_json = os.path.join(result_dir, "person_subtitle_mapping.json")
    aligned_json_in_subtitles = os.path.join(subtitles_dir, f"{base}_whisperx_speaker_aligned.json")

    if skip_done:
        status_data = _read_process_status(ws_dir)
        if status_data is not None:
            status = status_data.get("status")
            if status in PROCESS_STATUS_SKIP_VALUES:
                print(f"[INFO] {base} 已处理 (状态: {status})，跳过。")
                return base, "skipped"

    print(f"[INFO] 开始处理 {base}", flush=True)
    with pipeline_lock:
        try:
            vid_tracks, scores = pipeline.process_video(
                video_path=shot_path,
                workspace_base=asd_root,
                threshold=threshold,
            )
        except Exception as exc:
            _write_process_status(
                ws_dir,
                PROCESS_STATUS_FAILED,
                message=str(exc),
                extra={"error_type": "ASD_MAIN_FLOW"},
            )
            return base, f"ASD 主流程失败: {exc}"

    parse_asd_results(vid_tracks, scores, fps=fps, threshold=threshold)

    subtitle_input = srt_path or json_path
    aligned = []
    if subtitle_input:
        try:
            aligned = align_subtitles_with_speakers(
                vid_tracks,
                scores,
                subtitle_path=subtitle_input,
                word_json_path=json_path,
                fps=fps,
                threshold=threshold,
            )
        except Exception as exc:
            aligned = []

    try:
        with open(mapping_json, "w", encoding="utf-8") as f:
            json.dump(aligned, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        pass

    if aligned:
        try:
            with open(aligned_json_in_subtitles, "w", encoding="utf-8") as f:
                json.dump(aligned, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    with pipeline_lock:
        try:
            pipeline.export_speaker_portraits_yolo(shot_path, vid_tracks, portraits_dir)
        except Exception:
            pass

    if aligned and pipeline.last_track_mp4_paths:
        try:
            all_tracks_mp4 = pipeline.last_track_mp4_paths[0]
            audio_file_path = os.path.join(ws_dir, "pyavi", "audio.wav")
            pipeline.render_aligned_subtitles_on_video(
                input_video_path=all_tracks_mp4,
                audio_file_path=audio_file_path,
                aligned_subtitles=aligned,
                output_mp4_path=out_mp4,
                fps=fps,
            )
        except Exception:
            pass

    # 写入显式状态标记，解耦跳过逻辑与具体输出文件
    if not vid_tracks or not scores:
        asd_path = os.path.join(ws_dir, "asd_results.json")
        if os.path.isfile(asd_path):
            try:
                with open(asd_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("skip_reason") == "precheck_no_trackable_face":
                    pass  # 已在 process_video 中写入 NO_TRACKABLE_FACE
                else:
                    _write_process_status(
                        ws_dir,
                        PROCESS_STATUS_NO_VALID_TRACK,
                        message="无有效 track（场景过短或无人脸）",
                    )
            except Exception:
                _write_process_status(
                    ws_dir,
                    PROCESS_STATUS_NO_VALID_TRACK,
                    message="无有效 track",
                )
        else:
            _write_process_status(
                ws_dir,
                PROCESS_STATUS_NO_VALID_TRACK,
                message="无有效 track",
            )
    else:
        best_track, _ = pick_target_track_by_asd(
            vid_tracks, scores, fps=fps, threshold=threshold
        )
        if best_track is None:
            _write_process_status(
                ws_dir,
                PROCESS_STATUS_NO_VALID_SPEAKER,
                message="有 track 但无有效说话人（均低于阈值）",
            )
        else:
            _write_process_status(
                ws_dir,
                PROCESS_STATUS_COMPLETED,
                message="处理完成",
            )

    return base, "ok"


def run_asd_for_dataset(
    dataset_dir,
    pipeline=None,
    threshold=-0.4,
    skip_done=True,
    fps=25,
    n_gpus=None,
    min_duration_sec=2.0,
    shot_fps=24,
):
    """
    对 AVAGen 整个数据集执行 ASD，按 app.py 方式保存。
    支持多卡多线程：每张 GPU 一个模型实例，任务按轮询分配到各卡并行处理。

    输出：
    - asd/<shot>/pytracks/all_tracks_with_dialog.mp4
    - asd/<shot>/portraits_yolo/*.jpg
    - asd/<shot>/results/person_subtitle_mapping.json
    - asd/<shot>/tracking_results.json, asd_results.json
    - subtitles/<shot>_whisperx_speaker_aligned.json
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    all_targets = []
    movie_dirs = []
    for name in sorted(os.listdir(dataset_dir)):
        movie_dir = os.path.join(dataset_dir, name)
        if not os.path.isdir(movie_dir):
            continue
        targets = _collect_asd_targets(movie_dir, min_duration_sec=min_duration_sec, shot_fps=shot_fps)
        if targets:
            all_targets.extend([(movie_dir, name, t) for t in targets])
            movie_dirs.append((movie_dir, name))

    if not all_targets:
        print(f"[INFO] 数据集 {dataset_dir} 中无可处理片段（需 shots/、subtitles/、clips_check.jsonl 且 has_person=True、has_text=False、时长>=2s、有真实台词）")
        return

    n_gpus = n_gpus or (torch.cuda.device_count() if torch.cuda.is_available() else 0) or 1
    n_gpus = min(n_gpus, len(all_targets), torch.cuda.device_count() if torch.cuda.is_available() else 1)

    print(f"[INFO] 共 {len(movie_dirs)} 个电影目录，{len(all_targets)} 个待处理镜头，使用 {n_gpus} 张 GPU 并行")

    if n_gpus <= 1:
        # 单卡模式：使用传入的 pipeline 或默认创建
        pipe = pipeline or LightASDPipeline(device="cuda" if torch.cuda.is_available() else "cpu")
        lock = Lock()
        for movie_dir, movie_name, (shot_path, base, srt_path, json_path) in tqdm(all_targets, desc="ASD 批量处理"):
            task = (movie_dir, movie_name, (shot_path, base, srt_path, json_path))
            _process_one_target(pipe, lock, task, threshold, skip_done, fps)
    else:
        # 多卡模式：每卡一个模型，线程池并行
        pipelines = [LightASDPipeline(device=f"cuda:{i}") for i in range(n_gpus)]
        locks = [Lock() for _ in range(n_gpus)]

        def _run_task(idx_task):
            idx, task = idx_task
            pipe_idx = idx % n_gpus
            return _process_one_target(
                pipelines[pipe_idx],
                locks[pipe_idx],
                task,
                threshold,
                skip_done,
                fps,
            )

        tasks_with_idx = [(i, t) for i, t in enumerate(all_targets)]
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = {executor.submit(_run_task, item): item for item in tasks_with_idx}
            for future in tqdm(as_completed(futures), total=len(futures), desc="ASD 批量处理"):
                try:
                    base, status = future.result()
                    if status != "ok" and status != "skipped":
                        print(f"[WARN] {base}: {status}")
                except Exception as exc:
                    item = futures.get(future, (None, None))
                    base = item[1][1] if item[1] else "unknown"
                    print(f"[WARN] {base}: {exc}")

    print(f"[INFO] ASD 批量处理完成：{dataset_dir}")


def parse_asd_results(vidTracks, scores, fps=25, threshold=-0.5):
    print(f"\n========== 说话时间段分析 (阈值: {threshold}) ==========")
    all_track_segments = get_speaking_segments(vidTracks, scores, fps=fps, threshold=threshold)
    for track_idx, segments in enumerate(all_track_segments):
        if not segments:
            print(f"Track {track_idx:05d}: [全程未说话]")
        else:
            print(f"Track {track_idx:05d} 说话时间段:")
            for seg in segments:
                print(f"  - {seg['start_str']} 到 {seg['end_str']} (持续 {seg['duration']:.2f} 秒)")
    print("========================================================\n")

    best_track, rank_list = pick_target_track_by_asd(vidTracks, scores, fps=fps, threshold=threshold)
    if best_track is None:
        print("ASD 自动选人结果: 未找到有效轨迹。")
        return

    print("========== ASD 自动选人结果 ==========")
    print(
        f"目标人物: Track {best_track['track_idx']:05d} | "
        f"总说话时长: {best_track['total_speaking_duration']:.2f}s | "
        f"平均分: {best_track['mean_score']:.3f} | 峰值分: {best_track['peak_score']:.3f}"
    )
    if best_track["segments"]:
        print("目标人物说话时间段:")
        for seg in best_track["segments"]:
            print(f"  - {seg['start_str']} 到 {seg['end_str']} (持续 {seg['duration']:.2f} 秒)")
    else:
        print("目标人物在当前阈值下无有效说话片段。")

    print("候选排序(Top-3):")
    for i, item in enumerate(rank_list[:3], start=1):
        print(f"  {i}. Track {item['track_idx']:05d} | 时长 {item['total_speaking_duration']:.2f}s | 均分 {item['mean_score']:.3f} | 峰值 {item['peak_score']:.3f}")
    print("=====================================\n")

if __name__ == "__main__":
    import argparse

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "Datasets", "AVAGen")

    parser = argparse.ArgumentParser(description="Light-ASD 说话人检测（单视频或整数据集）")
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="输入视频路径；不传则处理整个 --dataset 目录",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help=f"数据集根目录（默认：{DEFAULT_DATASET_DIR}），与 app.py 输出结构一致",
    )
    parser.add_argument(
        "--no-skip-done",
        action="store_true",
        help="批量模式下不跳过已完成的镜头，重新处理",
    )
    parser.add_argument("--fps", type=float, default=25, help="视频帧率 (默认 25)")
    parser.add_argument("--threshold", type=float, default=-0.4, help="说话判定阈值")
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=None,
        help="数据集模式：使用的 GPU 数量，每卡一个模型并行。默认自动检测全部可用 GPU",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="数据集模式：跳过时长低于此值（秒）的视频，默认 2.0",
    )
    parser.add_argument("--srt", type=str, default=None, help="单视频模式：字幕路径(.srt/.json)，执行台词与说话人对齐")
    parser.add_argument("--word-json", type=str, default=None, help="单视频模式：whisperx 词级 json，配合 --srt 做句词融合对齐")
    args = parser.parse_args()

    pipeline = LightASDPipeline()

    if args.video:
        # 单视频模式
        if not os.path.isfile(args.video):
            print(f"错误：视频不存在: {args.video}")
            sys.exit(1)

        vidTracks, scores = pipeline.process_video(args.video, threshold=args.threshold)
        parse_asd_results(vidTracks, scores, fps=args.fps, threshold=args.threshold)

        if args.srt:
            aligned = align_subtitles_with_speakers(
                vidTracks,
                scores,
                subtitle_path=args.srt,
                word_json_path=args.word_json,
                fps=args.fps,
                threshold=args.threshold,
            )
            print("\n========== 最终台词与说话人清单 ==========")
            for item in aligned:
                speaker_name = item["speaker"].replace(" ", "_")
                print(f"[{speaker_name}] : {item['text']}")
            print("==========================================\n")

            video_name = os.path.splitext(os.path.basename(args.video))[0]
            portrait_dir = os.path.join("./asd_workspace", video_name, "portraits_yolo")
            pipeline.export_speaker_portraits_yolo(args.video, vidTracks, portrait_dir)

            if aligned and pipeline.last_track_mp4_paths:
                all_tracks_mp4 = pipeline.last_track_mp4_paths[0]
                ws_dir = os.path.dirname(os.path.dirname(all_tracks_mp4))
                audio_file_path = os.path.join(ws_dir, "pyavi", "audio.wav")
                output_with_dialog = os.path.splitext(all_tracks_mp4)[0] + "_with_dialog.mp4"
                pipeline.render_aligned_subtitles_on_video(
                    input_video_path=all_tracks_mp4,
                    audio_file_path=audio_file_path,
                    aligned_subtitles=aligned,
                    output_mp4_path=output_with_dialog,
                    fps=args.fps,
                )
    else:
        # 批量模式：处理整个 AVAGen 数据集，按 app.py 方式保存
        if not os.path.isdir(args.dataset):
            print(f"错误：数据集目录不存在: {args.dataset}")
            sys.exit(1)

        run_asd_for_dataset(
            dataset_dir=args.dataset,
            pipeline=pipeline,
            threshold=args.threshold,
            skip_done=not args.no_skip_done,
            fps=args.fps,
            n_gpus=args.n_gpus,
            min_duration_sec=args.min_duration,
        )