import os
import json
import glob
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# ================= CONFIGURATION =================
ROOT_DIR = "." 
OUTPUT_DIR = "./data/processed"
SAMPLES_DIR = "./training_data_samples"

FRAMES_PER_CLIP = 8 
TARGET_RES = (336, 336)
TRAIN_SUBJECTS = ["U0101", "U0102"]
VAL_SUBJECTS = ["U0107"]
TEST_SUBJECTS = ["U0108"]
OP_MAPPING = {
    "Assemble Box": "Box Setup",
    "Insert Items": "Inner Packing",
    "Tape": "Tape",
    "Attach Box Label": "Label",
    "Attach Shipping Label": "Label",
    "Scan Label": "Final Check",
    "Fill out Order": "Final Check",
    "Put on Back Table": "Pack",
    "Picking": "Inner Packing",
    "Relocate Item Label": "Label",
    "Idle": "Idle"
}

# ================= UTILS =================

def iso_to_unix_ms(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except:
        return 0

def load_coco_keypoints(json_path):
    if not os.path.exists(json_path): return {}
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        timestamp_index = {}
        annotations = data.get("annotations", [])
        for ann in annotations:
            ts = int(ann['image_id']) 
            kpts = ann['keypoints']
            timestamp_index[ts] = kpts
        return timestamp_index
    except:
        return {}

def calculate_motion_score(kpts1, kpts2):
    if not kpts1 or not kpts2: return 0.0
    k1 = np.array(kpts1).reshape(-1, 3)[:, :2]
    k2 = np.array(kpts2).reshape(-1, 3)[:, :2]
    v1 = np.array(kpts1).reshape(-1, 3)[:, 2] > 0.1
    v2 = np.array(kpts2).reshape(-1, 3)[:, 2] > 0.1
    mask = v1 & v2
    if not mask.any(): return 0.0
    return np.mean(np.linalg.norm(k1[mask] - k2[mask], axis=1))

def render_skeleton_frame(keypoints):
    img = np.zeros((336, 336, 3), dtype=np.uint8)
    if not keypoints: return img
    scale_x, scale_y = 336 / 1280, 336 / 720
    points = []
    data = np.array(keypoints).reshape(-1, 3)
    for x, y, v in data:
        if v > 0.1:
            points.append((int(x * scale_x), int(y * scale_y)))
        else:
            points.append(None)
    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
             (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    for idx_a, idx_b in edges:
        if idx_a < len(points) and idx_b < len(points):
            if points[idx_a] and points[idx_b]:
                cv2.line(img, points[idx_a], points[idx_b], (0, 255, 0), 2)
    return img

# ================= PIPELINE =================

def process_subject(subject_id, is_test=False):
    dataset = []
    subject_path = os.path.join(ROOT_DIR, subject_id)
    if not os.path.exists(subject_path): return []

    anno_files = glob.glob(os.path.join(subject_path, "annotation", "openpack-actions", "*.csv"))
    
    for csv_path in anno_files:
        session_id = os.path.basename(csv_path).split(".")[0]
        print(f"Processing {subject_id} {session_id}...")
        
        json_path = os.path.join(subject_path, "kinect", "2d-kpt", "mmpose-hrnet-w48-posetrack18-384x288-posewarper-stage2", "single", f"{session_id}.json")
        frame_db = load_coco_keypoints(json_path)
        if not frame_db: continue
        
        df = pd.read_csv(csv_path)
        available_ts = sorted(frame_db.keys())
        
        for i in range(len(df) - 1):
            row_curr = df.iloc[i]
            row_next = df.iloc[i+1]
            op_curr = OP_MAPPING.get(row_curr['operation'], "Unknown")
            op_next = OP_MAPPING.get(row_next['operation'], "Unknown")
            
            if op_curr == op_next: continue 
            
            boundary_ms = iso_to_unix_ms(row_curr['end'])
            start_ms = boundary_ms - 2500 
            end_ms = boundary_ms + 2500   
            
            window_ts = [t for t in available_ts if start_ms <= t <= end_ms]
            if len(window_ts) < FRAMES_PER_CLIP: continue
            
            scores = []
            for j in range(len(window_ts) - 1):
                score = calculate_motion_score(frame_db[window_ts[j]], frame_db[window_ts[j+1]])
                scores.append(score)
            
            chunk_size = len(window_ts) // FRAMES_PER_CLIP
            selected_ts = []
            for k in range(FRAMES_PER_CLIP):
                chunk = window_ts[k*chunk_size : (k+1)*chunk_size]
                if not chunk: continue
                best_t = chunk[0]
                max_m = -1
                for t_idx, t in enumerate(chunk):
                    global_idx = k*chunk_size + t_idx
                    if global_idx < len(scores) and scores[global_idx] > max_m:
                        max_m = scores[global_idx]
                        best_t = t
                selected_ts.append(best_t)

            if len(selected_ts) < FRAMES_PER_CLIP: continue

            clip_images = []
            for t in selected_ts:
                img = render_skeleton_frame(frame_db[t])
                fname = f"{session_id}_{i}_{t}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, fname), img)
                clip_images.append(fname)
                if len(os.listdir(SAMPLES_DIR)) < 20:
                    cv2.imwrite(os.path.join(SAMPLES_DIR, f"SAMPLE_{fname}"), img)

            dataset.append({
                "id": f"{session_id}_b{i}",
                "video": clip_images,
                "conversations": [
                    {"from": "user", "value": "Analyze this warehouse packaging video. Identify the current operation and predict the next."},
                    {"from": "assistant", "value": json.dumps({
                        "dominant_operation": op_curr,
                        "temporal_segment": {"start_frame": 0, "end_frame": 4},
                        "anticipated_next_operation": op_next,
                        "confidence": 0.95
                    })}
                ]
            })
    return dataset

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    
    # 1. Training Set
    train_data = []
    for sub in TRAIN_SUBJECTS:
        train_data += process_subject(sub)
    with open("training_data.json", "w") as f:
        json.dump(train_data, f, indent=2)
    print(f"Train samples: {len(train_data)}")

    # 2. Validation Set
    val_data = []
    for sub in VAL_SUBJECTS:
        val_data += process_subject(sub)
    with open("val_data.json", "w") as f:
        json.dump(val_data, f, indent=2)
    print(f"Val samples: {len(val_data)}")

    # 3. Test Set
    test_data = []
    for sub in TEST_SUBJECTS:
        test_data += process_subject(sub)
    with open("test_data.json", "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"Test samples: {len(test_data)}")

    print("Pipeline Complete with required splits.")

if __name__ == "__main__":
    main()