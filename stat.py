#!/usr/bin/env python3
"""
AVAGen 数据集统计脚本
统计：切片数量、时长、标注情况（无文本有人物、有台词、说话人数）
以及 Light-ASD 检测结果：说话人数量、是否全部台词匹配（无 UNK）
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def parse_clip_duration(filename: str, fps: float = 24.0) -> float | None:
    """从文件名解析时长（秒）。格式: scene_XXX_SSSSSS_EEEEEE.mp4，数字为帧索引"""
    m = re.match(r"scene_\d+_(\d+)_(\d+)\.mp4", filename)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        return (end - start) / fps  # 帧数 -> 秒
    return None


def load_clips_check(movie_dir: Path) -> list[dict]:
    """加载 clips_check.jsonl"""
    path = movie_dir / "clips_check.jsonl"
    if not path.exists():
        return []
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def analyze_subtitle(movie_dir: Path, clip_basename: str) -> tuple[bool, int]:
    """
    分析字幕：是否有台词、说话人数量
    返回 (has_dialogue, num_speakers)
    """
    sub_dir = movie_dir / "subtitles"
    json_path = sub_dir / f"{Path(clip_basename).stem}_whisperx.json"
    if not json_path.exists():
        return False, 0

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False, 0

    segments = data.get("segments", [])
    if not segments:
        return False, 0

    speakers = set()
    has_dialogue = False
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if text:
            has_dialogue = True
        spk = seg.get("speaker")
        if spk:
            speakers.add(spk)

    return has_dialogue, len(speakers)


def analyze_light_asd_aligned(movie_dir: Path, clip_basename: str) -> tuple[int | None, bool | None, int, int]:
    """
    分析 Light-ASD 对齐结果：说话人数量、是否全部台词匹配（无 UNK）
    返回 (num_speakers, all_matched, total_lines, unk_count)
    若文件不存在返回 (None, None, 0, 0)
    """
    sub_dir = movie_dir / "subtitles"
    json_path = sub_dir / f"{Path(clip_basename).stem}_whisperx_speaker_aligned.json"
    if not json_path.exists():
        return None, None, 0, 0

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None, 0, 0

    if not isinstance(data, list):
        return None, None, 0, 0

    speakers = set()
    unk_count = 0
    total_lines = 0
    for seg in data:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        total_lines += 1
        spk = seg.get("speaker") or seg.get("speaker_tag") or ""
        tag = seg.get("speaker_tag", "")
        if tag == "UNK" or spk in ("UNKNOWN", "UNK"):
            unk_count += 1
        else:
            speakers.add(spk or tag)

    num_speakers = len(speakers)
    all_matched = (unk_count == 0 and total_lines > 0) if total_lines > 0 else None
    return num_speakers, all_matched, total_lines, unk_count


def stat_movie(movie_dir: Path) -> dict:
    """统计单个电影"""
    clips_check = load_clips_check(movie_dir)
    shots_dir = movie_dir / "shots"

    if not clips_check:
        # 若无 clips_check，从 shots 目录推断
        clip_files = list(shots_dir.glob("*.mp4")) if shots_dir.exists() else []
        clips_check = [{"file": f.name, "has_text": None, "has_person": None} for f in clip_files]

    total_clips = len(clips_check)
    durations = []
    no_text_has_person = 0
    with_dialogue = 0
    speaker_counts = defaultdict(int)
    # Light-ASD 统计
    asd_clips = 0  # 有 Light-ASD 对齐结果的切片数
    asd_speaker_counts = defaultdict(int)  # 说话人数量分布
    asd_all_matched = 0  # 全部台词匹配（无 UNK）的切片数
    asd_has_unk = 0  # 含 UNK 的切片数
    asd_total_unk = 0  # UNK 台词总条数

    for item in clips_check:
        fname = item.get("file", "")
        has_text = item.get("has_text")
        has_person = item.get("has_person")

        # 时长
        dur = parse_clip_duration(fname)
        if dur is not None:
            durations.append(dur)

        # 无文本且有人物
        if has_text is False and has_person is True:
            no_text_has_person += 1

        # 字幕：有台词、说话人数
        has_dialogue, num_speakers = analyze_subtitle(movie_dir, fname)
        if has_dialogue:
            with_dialogue += 1
            speaker_counts[num_speakers] += 1

        # Light-ASD 对齐结果：说话人数量、是否全部匹配
        asd_num_spk, asd_all_m, asd_total, asd_unk = analyze_light_asd_aligned(movie_dir, fname)
        if asd_num_spk is not None:
            asd_clips += 1
            asd_speaker_counts[asd_num_spk] += 1
            if asd_all_m is True:
                asd_all_matched += 1
            if asd_unk > 0:
                asd_has_unk += 1
                asd_total_unk += asd_unk

    total_dur = sum(durations)
    avg_dur = total_dur / len(durations) if durations else 0
    min_dur = min(durations) if durations else 0
    max_dur = max(durations) if durations else 0

    return {
        "total_clips": total_clips,
        "total_duration_sec": total_dur,
        "avg_duration_sec": avg_dur,
        "min_duration_sec": min_dur,
        "max_duration_sec": max_dur,
        "no_text_has_person": no_text_has_person,
        "with_dialogue": with_dialogue,
        "speaker_distribution": dict(speaker_counts),
        "asd_clips": asd_clips,
        "asd_speaker_distribution": dict(asd_speaker_counts),
        "asd_all_matched": asd_all_matched,
        "asd_has_unk": asd_has_unk,
        "asd_total_unk": asd_total_unk,
    }


def format_duration(sec: float) -> str:
    """格式化时长为 HH:MM:SS"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    if h > 0:
        return f"{h}h {m}m {s:.1f}s"
    if m > 0:
        return f"{m}m {s:.1f}s"
    return f"{s:.1f}s"


def main():
    base = Path(__file__).resolve().parent / "Datasets" / "AVAGen"
    if not base.exists():
        print(f"目录不存在: {base}")
        return

    movie_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    if not movie_dirs:
        print("未找到电影目录")
        return

    all_stats = []
    for movie_dir in movie_dirs:
        name = movie_dir.name
        s = stat_movie(movie_dir)
        s["name"] = name
        all_stats.append(s)

    # 汇总
    total_clips = sum(x["total_clips"] for x in all_stats)
    total_dur = sum(x["total_duration_sec"] for x in all_stats)
    total_no_text_has_person = sum(x["no_text_has_person"] for x in all_stats)
    total_with_dialogue = sum(x["with_dialogue"] for x in all_stats)
    global_speaker_dist = defaultdict(int)
    for s in all_stats:
        for k, v in s["speaker_distribution"].items():
            global_speaker_dist[k] += v
    total_asd_clips = sum(x["asd_clips"] for x in all_stats)
    total_asd_all_matched = sum(x["asd_all_matched"] for x in all_stats)
    total_asd_has_unk = sum(x["asd_has_unk"] for x in all_stats)
    total_asd_unk_lines = sum(x["asd_total_unk"] for x in all_stats)
    global_asd_speaker_dist = defaultdict(int)
    for s in all_stats:
        for k, v in s["asd_speaker_distribution"].items():
            global_asd_speaker_dist[k] += v

    # 打印
    print("=" * 70)
    print("AVAGen 数据集统计")
    print("=" * 70)
    print(f"\n【汇总】")
    print(f"  电影数量:     {len(movie_dirs)}")
    print(f"  切片总数:     {total_clips}")
    print(f"  总时长:       {format_duration(total_dur)}")
    print(f"  无文本有人物: {total_no_text_has_person} (需补充字幕的切片)")
    print(f"  有台词:       {total_with_dialogue}")
    print(f"  说话人分布:   {dict(sorted(global_speaker_dist.items()))}")
    print(f"\n【Light-ASD 检测结果】")
    print(f"  有对齐结果:   {total_asd_clips} 切片")
    print(f"  说话人分布:   {dict(sorted(global_asd_speaker_dist.items()))}")
    print(f"  全部台词匹配: {total_asd_all_matched} 切片 (无 UNK)")
    print(f"  含 UNK 切片:  {total_asd_has_unk}   UNK 台词总条数: {total_asd_unk_lines}")

    all_durations = []
    for s in all_stats:
        n = s["total_clips"]
        avg = s["avg_duration_sec"]
        all_durations.extend([avg] * n)  # 近似
    # 更精确：从各电影 min/max 推断
    print(f"\n【切片时长】")
    if all_stats:
        durs = []
        for s in all_stats:
            # 用 avg * n 近似总时长已用于 total_dur，这里用 min/max/avg 展示
            durs.append(s["min_duration_sec"])
            durs.append(s["max_duration_sec"])
        global_min = min(s["min_duration_sec"] for s in all_stats if s["total_clips"] > 0)
        global_max = max(s["max_duration_sec"] for s in all_stats)
        global_avg = total_dur / total_clips if total_clips else 0
        print(f"  最短: {global_min:.2f}s  最长: {global_max:.2f}s  平均: {global_avg:.2f}s")

    print(f"\n【各电影明细】")
    print("-" * 70)
    for s in all_stats:
        print(f"\n  {s['name']}")
        print(f"    切片: {s['total_clips']}  时长: {format_duration(s['total_duration_sec'])}  "
              f"平均: {s['avg_duration_sec']:.2f}s")
        print(f"    无文本有人物: {s['no_text_has_person']}  有台词: {s['with_dialogue']}  "
              f"说话人分布: {s['speaker_distribution']}")
        print(f"    Light-ASD: {s['asd_clips']} 切片  说话人分布: {s['asd_speaker_distribution']}  "
              f"全部匹配: {s['asd_all_matched']}  含UNK: {s['asd_has_unk']} ({s['asd_total_unk']}条)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
