import os
import json
import glob
import shutil
from typing import Dict, Any, List

from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 原始数据根目录
VIDEO_ROOT = os.path.join(BASE_DIR, "Datasets", "AVAGen-480P")
CHAR_BBOX_ROOT = os.path.join(BASE_DIR, "Datasets", "AVAGen-480P-Character-bbox")
GROUNDED_CAPTION_ROOT = os.path.join(BASE_DIR, "Datasets", "AVAGen-480P-Grounded-Caption")
SPEAKER_ROOT = os.path.join(BASE_DIR, "Datasets", "AVAGen-480P-Speaker-Results")

# 过滤后数据集根目录
FILTERED_ROOT = os.path.join(BASE_DIR, "Datasets", "AVAGen-480P-Filtered")


def has_speaking_person(speaker_json: Dict[str, Any]) -> bool:
    """判断说话人标注中是否至少有一个人在说话。"""
    if isinstance(speaker_json, dict):
        for value in speaker_json.values():
            if isinstance(value, dict) and value.get("is_speaking"):
                return True
    elif isinstance(speaker_json, list):
        for item in speaker_json:
            if isinstance(item, dict) and item.get("is_speaking"):
                return True
    return False


def get_speaking_chars(speaker_json: Dict[str, Any]) -> List[str]:
    """返回正在说话的 char 列表（例如 ['<char_1>', '<char_2>']）。"""
    speaking: List[str] = []
    if isinstance(speaker_json, dict):
        for key, value in speaker_json.items():
            if isinstance(value, dict) and value.get("is_speaking"):
                speaking.append(key)
    elif isinstance(speaker_json, list):
        # 如果是列表，尝试从每个元素中读取 'char' 或 'char_id' 之类的字段
        for item in speaker_json:
            if isinstance(item, dict) and item.get("is_speaking"):
                char_key = item.get("char") or item.get("char_id")
                if char_key:
                    speaking.append(str(char_key))
    return speaking


def collect_norm_person_images(char_bbox_dir: str, clip_id: str) -> List[str]:
    """收集该 clip 下所有 *_person*_norm.jpg，并返回相对 movie 目录的路径列表。"""
    pattern = os.path.join(char_bbox_dir, f"{clip_id}_person*_norm.jpg")
    paths = sorted(glob.glob(pattern))
    return [os.path.basename(p) for p in paths]


def build_output_record(
    movie: str,
    clip_id: str,
    clip_meta: Dict[str, Any],
    grounded: Dict[str, Any],
    norm_images: List[str],
    speaking_chars: List[str],
) -> Dict[str, Any]:
    """构造输出 json 结构。"""
    return {
        "movie": movie,
        "clip_id": clip_id,
        "clip": clip_meta,
        "grounding_info": grounded.get("grounding_info"),
        "caption": grounded.get("rewritten_result"),
        "norm_person_images": norm_images,
        "speaking_chars": speaking_chars,
    }


def main() -> None:
    os.makedirs(FILTERED_ROOT, exist_ok=True)

    total = 0
    kept = 0

    # 遍历 AVAGen-480P-Speaker-Results 下所有电影和 clip_speaker.json
    for movie in tqdm(sorted(os.listdir(SPEAKER_ROOT)), desc="Movies"):
        movie_speaker_dir = os.path.join(SPEAKER_ROOT, movie)
        if not os.path.isdir(movie_speaker_dir):
            continue

        movie_video_dir = os.path.join(VIDEO_ROOT, movie)
        movie_char_dir = os.path.join(CHAR_BBOX_ROOT, movie)
        movie_grounded_dir = os.path.join(GROUNDED_CAPTION_ROOT, movie)

        if not (
            os.path.isdir(movie_video_dir)
            and os.path.isdir(movie_char_dir)
            and os.path.isdir(movie_grounded_dir)
        ):
            continue

        filtered_movie_dir = os.path.join(FILTERED_ROOT, movie)
        os.makedirs(filtered_movie_dir, exist_ok=True)

        speaker_files = [f for f in sorted(os.listdir(movie_speaker_dir)) if f.endswith("_speaker.json")]
        for fname in tqdm(speaker_files, desc=f"{movie} clips", leave=False):
            total += 1
            clip_id = fname.replace("_speaker.json", "")

            speaker_path = os.path.join(movie_speaker_dir, fname)
            try:
                with open(speaker_path, "r", encoding="utf-8") as f:
                    speaker_data = json.load(f)
            except Exception:
                continue

            # 条件 1：必须至少有一个人在说话
            if not has_speaking_person(speaker_data):
                continue

            speaking_chars = get_speaking_chars(speaker_data)
            if not speaking_chars:
                continue

            # 条件 2：必须要有 norm 的演员图片
            norm_images = collect_norm_person_images(movie_char_dir, clip_id)
            if not norm_images:
                continue

            video_json_path = os.path.join(movie_video_dir, f"{clip_id}.json")
            grounded_json_path = os.path.join(movie_grounded_dir, f"{clip_id}.json")
            video_mp4_path = os.path.join(movie_video_dir, f"{clip_id}.mp4")

            if not (os.path.isfile(video_json_path) and os.path.isfile(grounded_json_path)):
                continue

            try:
                with open(video_json_path, "r", encoding="utf-8") as f:
                    clip_meta = json.load(f)
                with open(grounded_json_path, "r", encoding="utf-8") as f:
                    grounded = json.load(f)
            except Exception:
                continue

            # 构造输出 json
            record = build_output_record(
                movie=movie,
                clip_id=clip_id,
                clip_meta=clip_meta,
                grounded=grounded,
                norm_images=norm_images,
                speaking_chars=speaking_chars,
            )

            out_json_path = os.path.join(filtered_movie_dir, f"{clip_id}.json")
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            # 拷贝 mp4（如果存在）
            if os.path.isfile(video_mp4_path):
                shutil.copy2(video_mp4_path, os.path.join(filtered_movie_dir, f"{clip_id}.mp4"))

            # 拷贝 norm person 图像
            for img_name in norm_images:
                src_img = os.path.join(movie_char_dir, img_name)
                if os.path.isfile(src_img):
                    shutil.copy2(src_img, os.path.join(filtered_movie_dir, img_name))

            kept += 1

    print(f"Total candidates: {total}, kept: {kept}")


if __name__ == "__main__":
    main()

