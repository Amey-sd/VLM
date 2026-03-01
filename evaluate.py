import os
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import gc

# --- CONFIGURATION ---
TEST_DATA_PATH = "./test_data.json"
IMAGE_FOLDER = "./data/processed"
ADAPTER_PATH = "./output_qwen_lora/final_adapter"
BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
RESULTS_FILE = "results.json"
NUM_TEST_SAMPLES = 30

# --- METRICS UTILS ---
def calculate_iou(pred_start, pred_end, gt_start, gt_end):
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    if inter_end <= inter_start: return 0.0
    inter_area = inter_end - inter_start
    
    pred_area = pred_end - pred_start
    gt_area = gt_end - gt_start
    union_area = pred_area + gt_area - inter_area
    
    if union_area <= 0: return 0.0
    return inter_area / union_area

def robust_json_parse(text):
    """
    Aggressively attempts to find and parse JSON in model output.
    """
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
        
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    
    try:
        return json.loads(text)
    except:
        try:
            fixed_text = text.replace("'", '"')
            return json.loads(fixed_text)
        except:
            return None

# --- INFERENCE ENGINE ---
def run_evaluation(model, processor, test_data, device, debug_name="Model"):
    results = {
        "correct_ops": 0,
        "iou_scores": [],
        "anticipation_correct": 0,
        "total": 0
    }
    
    print(f"Evaluating {debug_name} on {len(test_data)} samples...")
    
    for i, item in enumerate(tqdm(test_data)):
        image_paths = [os.path.join(IMAGE_FOLDER, img) for img in item["video"]]
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        
        if len(valid_paths) > 4:
            indices = np.linspace(0, len(valid_paths)-1, 4, dtype=int)
            valid_paths = [valid_paths[i] for i in indices]

        if not valid_paths: continue

        prompt = (
            "Analyze this warehouse video. Identify the action (e.g., 'Tape', 'Box Setup', 'Pack').\n"
            "Output strictly valid JSON with these exact keys:\n"
            "{\n"
            '  "dominant_operation": "String",\n'
            '  "temporal_segment": {"start_frame": 0, "end_frame": 8},\n'
            '  "anticipated_next_operation": "String",\n'
            '  "confidence": 1.0\n'
            "}"
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": valid_paths,
                        "fps": 1.0,
                        "min_pixels": 224 * 224, 
                        "max_pixels": 224 * 224,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        pred = robust_json_parse(output_text)
        gt = json.loads(item["conversations"][1]["value"])
        
        if i < 3:
            print(f"\n[{debug_name} Sample {i}]")
            print(f"RAW OUTPUT: {output_text}")
            print(f"PARSED: {pred}")
            print(f"GROUND TRUTH: {gt['dominant_operation']}")

        if pred is None:
            results["total"] += 1
            results["iou_scores"].append(0.0)
            continue

        pred_op = pred.get("dominant_operation", "Unknown")
        gt_op = gt.get("dominant_operation", "Unknown")
        if pred_op.lower() == gt_op.lower():
            results["correct_ops"] += 1
            
        pred_next = pred.get("anticipated_next_operation", "Unknown")
        gt_next = gt.get("anticipated_next_operation", "Unknown")
        if pred_next.lower() == gt_next.lower():
            results["anticipation_correct"] += 1
            
        p_seg = pred.get("temporal_segment", {})
        g_seg = gt.get("temporal_segment", {})
        iou = calculate_iou(
            p_seg.get("start_frame", 0), p_seg.get("end_frame", 0),
            g_seg.get("start_frame", 0), g_seg.get("end_frame", 8)
        )
        results["iou_scores"].append(iou)
        results["total"] += 1

    metrics = {
        "OCA": results["correct_ops"] / results["total"] if results["total"] else 0,
        "tIoU@0.5": sum(1 for x in results["iou_scores"] if x >= 0.5) / results["total"] if results["total"] else 0,
        "AA@1": results["anticipation_correct"] / results["total"] if results["total"] else 0
    }
    
    return {k: round(v, 2) for k, v in metrics.items()}

# --- MAIN DRIVER ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on {device}")

    try:
        with open(TEST_DATA_PATH, 'r') as f:
            full_data = json.load(f)
        test_data = full_data[-NUM_TEST_SAMPLES:]
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    final_metrics = {}

    # 1. EVALUATE BASE MODEL
    print("\n--- Evaluating Base Model ---")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, min_pixels=224*224, max_pixels=224*224)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    
    final_metrics["base_model"] = run_evaluation(model, processor, test_data, device, "Base")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. EVALUATE FINE-TUNED MODEL
    print("\n--- Evaluating Fine-Tuned Model ---")
    
    if os.path.exists(ADAPTER_PATH):
        try:
            # Reload Base
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
            )
            # Load Adapter
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)
            
            final_metrics["finetuned_model"] = run_evaluation(model, processor, test_data, device, "FineTuned")
        except Exception as e:
            print(f"Error loading adapter: {e}")
            final_metrics["finetuned_model"] = {"error": str(e)}
    else:
        print("Adapter not found! Skipping fine-tuned evaluation.")
        final_metrics["finetuned_model"] = {"error": "Adapter path invalid"}

    # 3. SAVE RESULTS
    print("\nFinal Results:")
    print(json.dumps(final_metrics, indent=2))
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(final_metrics, f, indent=2)

if __name__ == "__main__":
    main()