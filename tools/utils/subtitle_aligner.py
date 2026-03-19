"""
字幕逻辑：WhisperX / SRT 解析、时间轴 overlap 计算、align_subtitles_with_speakers。
"""

import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np


def format_time(seconds: float) -> str:
    """格式化为 MM:SS.ss"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def clean_subtitle_text(text: str) -> str:
    """清理字幕文本中的 SPEAKER 前缀等。"""
    cleaned = str(text).strip()
    cleaned = re.sub(r"^\[\s*SPEAKER[_\-\s]*\d+\s*\]\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^SPEAKER[_\-\s]*\d+\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\[\s*UNKNOWN\s*\]\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^UNKNOWN\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _parse_srt_timecode(timecode: str) -> float:
    parts = timecode.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"非法 SRT 时间戳: {timecode}")
    hours = int(parts[0])
    mins = int(parts[1])
    sec_parts = parts[2].split(",")
    if len(sec_parts) != 2:
        raise ValueError(f"非法 SRT 时间戳: {timecode}")
    secs = int(sec_parts[0])
    millis = int(sec_parts[1])
    return hours * 3600 + mins * 60 + secs + millis / 1000.0


def parse_srt_entries(srt_path: str) -> List[dict]:
    """解析 SRT 文件，返回 [{idx, start, end, text}, ...]"""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    blocks = re.split(r"\n\s*\n", content)
    entries = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        seq_line = lines[0]
        time_line = lines[1] if "-->" in lines[1] else lines[0]
        text_start_idx = 2 if "-->" in lines[1] else 1
        if "-->" not in time_line:
            continue
        start_tc, end_tc = [p.strip() for p in time_line.split("-->", 1)]
        try:
            start_sec = _parse_srt_timecode(start_tc)
            end_sec = _parse_srt_timecode(end_tc)
        except ValueError:
            continue
        if end_sec <= start_sec:
            continue
        text = clean_subtitle_text(" ".join(lines[text_start_idx:]).strip())
        if not text:
            continue
        try:
            idx = int(seq_line)
        except ValueError:
            idx = len(entries) + 1
        entries.append({"idx": idx, "start": round(start_sec, 3), "end": round(end_sec, 3), "text": text})
    return entries


def parse_whisperx_word_entries(json_path: str) -> List[dict]:
    """解析 WhisperX JSON，返回词级 [{idx, start, end, text}, ...]"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = []
    if isinstance(data, dict) and isinstance(data.get("word_segments"), list):
        words = data.get("word_segments", [])
    elif isinstance(data, dict) and isinstance(data.get("segments"), list):
        for seg in data.get("segments", []):
            if isinstance(seg, dict) and isinstance(seg.get("words"), list):
                words.extend(seg.get("words", []))
    entries = []
    for i, w in enumerate(words, start=1):
        if not isinstance(w, dict):
            continue
        start = w.get("start")
        end = w.get("end")
        text = clean_subtitle_text(str(w.get("word", "")).strip())
        if start is None or end is None or not text:
            continue
        start = float(start)
        end = float(end)
        if end <= start:
            continue
        entries.append({"idx": i, "start": round(start, 3), "end": round(end, 3), "text": text})
    return entries


def parse_subtitle_entries(subtitle_path: str) -> Tuple[List[dict], str]:
    """解析字幕文件，返回 (entries, granularity)，granularity 为 'word' 或 'sentence'。"""
    ext = os.path.splitext(subtitle_path)[1].lower()
    if ext == ".json":
        return parse_whisperx_word_entries(subtitle_path), "word"
    return parse_srt_entries(subtitle_path), "sentence"


def _overlap_duration(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    left = max(start_a, start_b)
    right = min(end_a, end_b)
    return max(0.0, right - left)


def infer_word_json_from_srt(srt_path: str) -> Optional[str]:
    """根据 SRT 路径推断同名的 WhisperX JSON 路径。"""
    candidate = os.path.splitext(srt_path)[0] + ".json"
    if os.path.isfile(candidate):
        return candidate
    return None


def _assign_entry_speaker(
    entry: dict,
    track_segments: List[List[dict]],
    min_overlap_sec: float = 0.05,
    min_overlap_ratio: float = 0.2,
) -> dict:
    """为单条字幕分配说话人 track。"""
    sub_start = entry["start"]
    sub_end = entry["end"]
    sub_dur = max(1e-6, sub_end - sub_start)

    best_track = None
    best_overlap = 0.0
    all_overlaps = []
    for track_idx, segs in enumerate(track_segments):
        overlap = 0.0
        for seg in segs:
            overlap += _overlap_duration(sub_start, sub_end, seg["start"], seg["end"])
        all_overlaps.append((track_idx, overlap))
        if overlap > best_overlap:
            best_overlap = overlap
            best_track = track_idx

    confidence = best_overlap / sub_dur
    if best_track is None or best_overlap < min_overlap_sec or confidence <= min_overlap_ratio:
        speaker_tag = "UNK"
        speaker = "UNKNOWN"
        speaker_track = None
    else:
        speaker_tag = f"T{best_track:02d}"
        speaker = f"Track {best_track:05d}"
        speaker_track = best_track

    top2 = sorted(all_overlaps, key=lambda x: x[1], reverse=True)[:2]
    return {
        "speaker_tag": speaker_tag,
        "speaker": speaker,
        "speaker_track_idx": speaker_track,
        "overlap_sec": round(best_overlap, 3),
        "confidence": round(confidence, 3),
        "top2_candidates": [{"track_idx": idx, "overlap_sec": round(ov, 3)} for idx, ov in top2],
    }


def get_speaking_segments(
    vidTracks: list,
    scores: list,
    fps: float = 25,
    threshold: float = -0.5,
    min_duration: float = 0.2,
) -> List[List[dict]]:
    """根据 ASD 分数计算每个 track 的说话时间段。"""
    all_track_segments = []
    for track_info, track_scores in zip(vidTracks, scores):
        frames = track_info["track"]["frame"]
        min_len = min(len(frames), len(track_scores))
        is_speaking = False
        start_time = 0.0
        raw_segments = []

        for i in range(min_len):
            score = track_scores[i]
            frame_num = frames[i]
            current_time = frame_num / fps
            if score >= threshold and not is_speaking:
                is_speaking = True
                start_time = current_time
            elif score < threshold and is_speaking:
                is_speaking = False
                raw_segments.append((start_time, current_time))

        if is_speaking:
            raw_segments.append((start_time, frames[min_len - 1] / fps))

        segments = []
        for start, end in raw_segments:
            duration = end - start
            if duration > min_duration:
                segments.append({
                    "start": round(start, 2), "end": round(end, 2),
                    "start_str": format_time(start), "end_str": format_time(end),
                    "duration": round(duration, 2),
                })
        all_track_segments.append(segments)
    return all_track_segments


def pick_target_track_by_asd(
    vidTracks: list,
    scores: list,
    fps: float = 25,
    threshold: float = -0.5,
    min_duration: float = 0.2,
) -> Tuple[Optional[dict], List[dict]]:
    """根据 ASD 分数选出最佳说话人 track，返回 (best_track, rank_list)。"""
    if not vidTracks or not scores:
        return None, []
    all_track_segments = get_speaking_segments(
        vidTracks, scores, fps=fps, threshold=threshold, min_duration=min_duration
    )
    candidates = []
    for track_idx, (track_info, track_scores, segments) in enumerate(
        zip(vidTracks, scores, all_track_segments)
    ):
        frames = track_info["track"]["frame"]
        min_len = min(len(frames), len(track_scores))
        if min_len <= 0:
            continue
        clipped_scores = np.array(track_scores[:min_len], dtype=np.float32)
        total_speaking_duration = sum(seg["duration"] for seg in segments)
        mean_score = float(np.mean(clipped_scores))
        peak_score = float(np.max(clipped_scores))
        candidates.append({
            "track_idx": track_idx,
            "total_speaking_duration": round(total_speaking_duration, 3),
            "mean_score": round(mean_score, 4),
            "peak_score": round(peak_score, 4),
            "segments": segments,
        })
    if not candidates:
        return None, []
    candidates.sort(
        key=lambda x: (x["total_speaking_duration"], x["mean_score"], x["peak_score"]),
        reverse=True,
    )
    return candidates[0], candidates


def align_subtitles_with_speakers(
    vidTracks: list,
    scores: list,
    subtitle_path: str,
    word_json_path: Optional[str] = None,
    fps: float = 25,
    threshold: float = -0.5,
    min_overlap_sec: float = 0.05,
    min_overlap_ratio: float = 0.2,
    sentence_vote_ratio: float = 0.5,
    write_output_json: bool = True,
) -> List[dict]:
    """
    将字幕与说话人 track 对齐。
    若 write_output_json 为 True，会写入 subtitle_path 同目录的 _speaker_aligned.json。
    """
    if not os.path.isfile(subtitle_path):
        print(f"[WARN] 字幕文件不存在: {subtitle_path}")
        return []

    subtitles, granularity = parse_subtitle_entries(subtitle_path)
    if not subtitles:
        print(f"[WARN] 未从字幕解析到有效内容: {subtitle_path}")
        return []

    track_segments = get_speaking_segments(
        vidTracks, scores, fps=fps, threshold=threshold, min_duration=0.0
    )

    fusion_words = None
    if granularity == "sentence":
        resolved_word_json = word_json_path or infer_word_json_from_srt(subtitle_path)
        if resolved_word_json and os.path.isfile(resolved_word_json):
            fusion_words = parse_whisperx_word_entries(resolved_word_json)
            print(f"[INFO] 启用句词融合对齐，词级来源: {resolved_word_json}")

    aligned = []
    print(f"\n========== 台词-说话人对齐结果 ({granularity}) ==========")
    for sub in subtitles:
        sub_start = sub["start"]
        sub_end = sub["end"]

        assign = _assign_entry_speaker(
            sub, track_segments, min_overlap_sec=min_overlap_sec, min_overlap_ratio=min_overlap_ratio
        )
        source = "sentence_overlap"

        if fusion_words:
            word_votes = {}
            word_hits = 0
            total_word_dur = 0.0
            for w in fusion_words:
                ov = _overlap_duration(sub_start, sub_end, w["start"], w["end"])
                if ov <= 0:
                    continue
                word_hits += 1
                total_word_dur += ov
                w_assign = _assign_entry_speaker(
                    w, track_segments, min_overlap_sec=min_overlap_sec * 0.5, min_overlap_ratio=0.0
                )
                tag = w_assign["speaker_tag"]
                if tag == "UNK":
                    continue
                word_votes[tag] = word_votes.get(tag, 0.0) + ov

            if word_votes and total_word_dur > 0:
                best_tag, best_dur = sorted(word_votes.items(), key=lambda x: x[1], reverse=True)[0]
                vote_ratio = best_dur / total_word_dur
                if vote_ratio > sentence_vote_ratio:
                    track_idx = int(best_tag[1:])
                    assign["speaker_tag"] = best_tag
                    assign["speaker"] = f"Track {track_idx:05d}"
                    assign["speaker_track_idx"] = track_idx
                    assign["confidence"] = round(vote_ratio, 3)
                    source = "word_vote"

        display_text = f"{assign['speaker_tag']}: {sub['text']}"
        top2_str = ", ".join([f"{x['track_idx']:05d}:{x['overlap_sec']:.2f}s" for x in assign["top2_candidates"]])

        print(
            f"[{sub['idx']:03d}] {format_time(sub_start)} -> {format_time(sub_end)} | "
            f"{assign['speaker_tag']} ({assign['speaker']}) | overlap={assign['overlap_sec']:.2f}s "
            f"({assign['confidence']:.0%}) | src={source} | {sub['text']}"
        )
        print(f"      候选Top2: {top2_str}")

        aligned.append({
            "subtitle_idx": sub["idx"],
            "start": sub_start,
            "end": sub_end,
            "text": sub["text"],
            "speaker": assign["speaker"],
            "speaker_tag": assign["speaker_tag"],
            "speaker_track_idx": assign["speaker_track_idx"],
            "display_text": display_text,
            "overlap_sec": assign["overlap_sec"],
            "confidence": assign["confidence"],
            "top2_candidates": assign["top2_candidates"],
            "assign_source": source,
        })
    print("========================================\n")

    if write_output_json:
        output_json = os.path.splitext(subtitle_path)[0] + "_speaker_aligned.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(aligned, f, ensure_ascii=False, indent=2)
        print(f"已写出台词-说话人对齐结果: {output_json}")
    return aligned
