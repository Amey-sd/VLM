from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
import tempfile
import shutil
import json
import os
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch
from qwen_vl_utils import process_vision_info

app = FastAPI()

print("Loading Qwen Model from transformers")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16 if device == "cuda" else "auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

print("Model loaded successfully")

def clean_json_output(text: str):
    """
    LLMs often wrap JSON in markdown like ```json ... ```.
    This function strips that out to parse the raw JSON.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

@app.get("/")
async def main():
    with open("templates/home.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    # A. Save uploaded video to a temporary file
    # Decord (used internally by Qwen) requires a file path, not bytes.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        shutil.copyfileobj(file.file, temp_video)
        temp_video_path = temp_video.name

    try:
        # B. Construct the Prompt
        # We explicitly ask for the JSON schema required by the assignment.
        prompt_text = (
            "Analyze this warehouse packaging video. Identify the current operation.\n"
            "Output strictly valid JSON with these keys:\n"
            "- dominant_operation (String: e.g., 'Tape', 'Pack', 'Label')\n"
            "- temporal_segment (Object: start_frame, end_frame)\n"
            "- anticipated_next_operation (String)\n"
            "- confidence (Float 0-1)\n"
            "Do not provide explanations, only the JSON object."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": temp_video_path,
                        # Hints for efficient decoding (sample frames)
                        "max_pixels": 360 * 420, 
                        "fps": 1.0, # Sample 1 frame per second to save memory/time
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # C. Prepare Inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # D. Run Inference
        # max_new_tokens=128 is enough for a small JSON object
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        
        # Trim the input tokens from the output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # E. Parse Output
        cleaned_json = clean_json_output(output_text)
        
        try:
            result_data = json.loads(cleaned_json)
            # Inject the clip ID as requested
            result_data["clip_id"] = file.filename 
        except json.JSONDecodeError:
            # Fallback if model outputs bad JSON (common in early phases)
            result_data = {
                "clip_id": file.filename,
                "error": "Model failed to generate valid JSON",
                "raw_output": output_text
            }

        return result_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # F. Cleanup: Delete the temp file to save disk space
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)