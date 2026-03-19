import argparse
import json
import logging
import os
import re
import warnings
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from qwen_omni_utils import process_mm_info


INPUT_ROOT = Path("Datasets/AVAGen")
OUTPUT_ROOT = Path("Datasets/AVAGen-QwenOmni-Caption")
FAILED_LOG_DIR = OUTPUT_ROOT / "_failed_logs"
MODEL_ID = "Models/Qwen2.5-Omni-3B"
DEFAULT_DEVICE = "cuda:0"


warnings.filterwarnings("ignore", category=UserWarning, module="qwen_omni_utils.v2_5.audio_process")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")
logging.getLogger("root").setLevel(logging.ERROR)


SCHEMA_TEXT = """
{
  "video_id": "string",
  "duration_sec": float,
  "shot": {
    "shot_type": "close_up | medium | full | over_shoulder | wide | insert | mixed | unknown",
    "camera_motion": "static | pan | tilt | push_in | pull_out | handheld | tracking | zoom | mixed | unknown",
    "framing": "string",
    "pace": "slow | medium | fast | mixed | unknown",
    "visual_focus": "string"
  },
  "scene": {
    "location": "string",
    "time_of_day": "string",
    "main_event": "string",
    "event_steps": ["string"],
    "props_or_objects": ["string"],
    "style_or_mood": "string"
  },
  "characters": [
    {
      "id": "<T00>",
      "visibility": "full | upper_body | face_only | partial | back_view | unknown",
      "actions": ["string"],
      "expression": "neutral | happy | angry | worried | sad | surprised | tense | contempt | mixed | unknown",
      "tone": "calm | excited | cold | sarcastic | threatening | pleading | nervous | firm | mixed | unknown",
      "attention_target": "<T01> | object | unknown",
      "interaction_state": "solo | listening | speaking_to_<T01> | confronting_<T01> | assisting_<T01> | following_<T01> | unknown"
    }
  ],
  "interactions": [
    {
      "subject": "<T00>",
      "object": "<T01>",
      "type": "dialogue | gaze | physical_contact | pursuit | cooperation | conflict | emotional_response | unknown",
      "description": "string",
      "intensity": "low | medium | high | unknown"
    }
  ],
  "caption": {
    "brief": "string",
    "dense": "string"
  }
}
""".strip()


PROMPT_TEXT = f"""
Task: Analyze the input video and produce a structured caption in strict JSON only.

Hard rules:
1. Output ONLY valid JSON. No markdown blocks. No extra commentary.
2. Every human character must be referenced only by identifiers like <T00>, <T01>, <T02>.
3. Never use generic references such as "man", "woman", "person", "someone", "people", "character" in the final JSON fields when referring to a tracked person. Use <Txx> instead.
4. Keep character identifiers consistent across "characters", "interactions", and "caption".
5. If a fact is unclear, use "unknown" instead of inventing details.
6. Describe camera motion, visible content, character interactions, expression, and speaking tone when inferable.
7. The "caption.brief" should be one sentence. The "caption.dense" should be 2-4 sentences.
8. The "event_steps", "actions", "props_or_objects", and "interactions" should stay concise and concrete.

Target JSON schema:
{SCHEMA_TEXT}
""".strip()


def build_conversation(video_path: Path):
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are an expert multimodal video captioning AI. You output strict JSON only."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        },
    ]


def clean_and_parse_json(raw_text: str):
    clean_text = raw_text.strip()
    if "assistant" in clean_text:
        clean_text = clean_text.split("assistant", 1)[-1].strip()
    clean_text = clean_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", clean_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
    return None


def get_video_duration_sec(video_path: Path):
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps > 0 and frame_count > 0:
            return round(frame_count / fps, 3)
    except Exception:
        pass
    return None


class OmniCaptioner:
    def __init__(self, model_id=MODEL_ID, device=DEFAULT_DEVICE):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None

    def ensure_loaded(self):
        if self.model is not None and self.processor is not None:
            return

        print(f"[QwenOmniCaption] Loading model on {self.device} from {self.model_id}...")
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=self.device if self.device != "cpu" else "cpu",
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_id)
        print("[QwenOmniCaption] Model loaded.")

    def analyze(self, video_path: Path, max_new_tokens=1024):
        self.ensure_loaded()

        conversation = build_conversation(video_path)
        use_audio_in_video = True

        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios, images, videos = process_mm_info(
            conversation,
            use_audio_in_video=use_audio_in_video,
        )

        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                use_audio_in_video=use_audio_in_video,
                return_audio=False,
                max_new_tokens=max_new_tokens,
            )

        raw_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return raw_text


def normalize_result(video_path: Path, parsed_json: dict):
    result = dict(parsed_json or {})
    result.setdefault("video_id", video_path.stem)
    result.setdefault("duration_sec", get_video_duration_sec(video_path) or "unknown")
    result.setdefault("shot", {})
    result.setdefault("scene", {})
    result.setdefault("characters", [])
    result.setdefault("interactions", [])
    result.setdefault("caption", {})
    return result


def process_single_video(
    video_path: Path,
    captioner: OmniCaptioner,
    input_root: Path,
    output_root: Path,
    failed_log_dir: Path,
    overwrite=False,
    max_new_tokens=1024,
):
    relative_path = video_path.relative_to(input_root)
    output_json_path = output_root / relative_path.with_suffix(".json")
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    failed_log_dir.mkdir(parents=True, exist_ok=True)

    if output_json_path.exists() and output_json_path.stat().st_size > 0 and not overwrite:
        return {"status": "skipped", "file": str(relative_path), "output": str(output_json_path)}

    raw_result = captioner.analyze(video_path, max_new_tokens=max_new_tokens)
    parsed_json = clean_and_parse_json(raw_result)

    status = "success"
    if parsed_json is None:
        status = "parse_error"
        failed_log_path = failed_log_dir / f"{video_path.stem}.txt"
        with open(failed_log_path, "w", encoding="utf-8") as f:
            f.write(raw_result)
        final_data = {
            "file_path": str(relative_path),
            "status": status,
            "error": "json_parse_failed",
            "raw_output_log": str(failed_log_path),
            "analysis": {},
        }
    else:
        final_data = {
            "file_path": str(relative_path),
            "status": status,
            "analysis": normalize_result(video_path, parsed_json),
        }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    return {"status": status, "file": str(relative_path), "output": str(output_json_path)}


def parse_args():
    parser = argparse.ArgumentParser(description="Use Qwen Omni to produce structured video captions.")
    parser.add_argument("-i", "--input", type=str, help="Single video path for manual testing.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output JSON path for single-video mode.")
    parser.add_argument("--input-root", type=str, default=str(INPUT_ROOT), help="Batch input root.")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT), help="Batch output root.")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--glob", type=str, default="*.mp4", help="Batch file glob under input root.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output JSON files.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N videos in batch mode.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Generation max_new_tokens.")
    return parser.parse_args()


def main():
    args = parse_args()

    captioner = OmniCaptioner(device=args.device)

    if args.input:
        video_path = Path(args.input)
        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        output_path = Path(args.output) if args.output else video_path.with_suffix(".caption.json")
        raw_result = captioner.analyze(video_path, max_new_tokens=args.max_new_tokens)
        parsed_json = clean_and_parse_json(raw_result)

        if parsed_json is None:
            print(raw_result)
            raise RuntimeError("Failed to parse model output as JSON.")

        final_data = {
            "file_path": str(video_path),
            "status": "success",
            "analysis": normalize_result(video_path, parsed_json),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        print(f"Saved structured caption to: {output_path}")
        return

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    failed_log_dir = output_root / "_failed_logs"
    failed_log_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    all_files = sorted(input_root.rglob(args.glob))
    if args.limit > 0:
        all_files = all_files[:args.limit]

    print(f"Found {len(all_files)} video files under {input_root}")
    results = []
    for video_path in tqdm(all_files, desc="Captioning"):
        results.append(
            process_single_video(
                video_path=video_path,
                captioner=captioner,
                input_root=input_root,
                output_root=output_root,
                failed_log_dir=failed_log_dir,
                overwrite=args.overwrite,
                max_new_tokens=args.max_new_tokens,
            )
        )

    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    parse_error_count = sum(1 for r in results if r["status"] == "parse_error")
    print(f"Finished. success={success_count}, skipped={skipped_count}, parse_error={parse_error_count}")


if __name__ == "__main__":
    main()
