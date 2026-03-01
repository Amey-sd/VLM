import os
import shutil
import tempfile
import json
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import PeftModel

app = FastAPI()

# --- CONFIGURATION ---
BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_PATH = "./output_qwen_lora/final_adapter"

print(f"Loading Base Model: {BASE_MODEL_ID}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 1. Load Base Model
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else "auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

# 2. Load Adapter (if exists)
has_adapter = False
if os.path.exists(ADAPTER_PATH):
    print(f"Loading LoRA Adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    has_adapter = True
else:
    print("No adapter found. Running in Base Model only mode.")
    model = base_model

print("System Ready.")

def clean_json_output(text: str):
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    return text.strip()

@app.get("/")
async def main():
    with open("templates/home.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form("finetuned")
):
    # A. Save Video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        shutil.copyfileobj(file.file, temp_video)
        temp_video_path = temp_video.name

    try:
        # B. Prepare Inputs
        prompt_text = (
            "Analyze this warehouse packaging video. Identify the current operation.\n"
            "Output strictly valid JSON with these keys:\n"
            "- dominant_operation (String: e.g., 'Tape', 'Pack', 'Label')\n"
            "- temporal_segment (Object: start_frame, end_frame)\n"
            "- anticipated_next_operation (String)\n"
            "- confidence (Float 0-1)\n"
        )

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": temp_video_path,
                    "max_pixels": 360 * 420, 
                    "fps": 1.0, 
                },
                {"type": "text", "text": prompt_text},
            ],
        }]

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
        
        if model_type == "base" and has_adapter:
            print("Running Inference: BASE MODEL (Adapter Disabled)")
            with model.disable_adapter():
                generated_ids = model.generate(**inputs, max_new_tokens=128)
        else:
            print(f"Running Inference: {'FINE-TUNED' if has_adapter else 'BASE (Fallback)'}")
            generated_ids = model.generate(**inputs, max_new_tokens=128)

        # D. Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # E. Parse
        cleaned_json = clean_json_output(output_text)
        try:
            result_data = json.loads(cleaned_json)
            result_data["clip_id"] = file.filename
            result_data["model_used"] = model_type
        except json.JSONDecodeError:
            result_data = {
                "clip_id": file.filename,
                "model_used": model_type,
                "error": "Model output invalid JSON",
                "raw_output": output_text
            }

        return result_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)