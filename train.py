import os
import yaml
import json
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

# --- 1. CONFIG LOAD ---
with open("finetune_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# --- 2. DATASET CLASS ---
class OpenPackVideoDataset(Dataset):
    def __init__(self, data_path, image_folder, processor):
        self.root = image_folder
        self.processor = processor
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            item = self.data[idx]
            
            # 1. Get Image Paths
            image_paths = [os.path.join(self.root, img) for img in item["video"]]
            
            # 2. Validation & Sampling
            valid_paths = [p for p in image_paths if os.path.exists(p)]
            
            # FIX: Ensure we only take exactly 'frames_per_clip' (4)
            target_frames = cfg["frames_per_clip"]
            if len(valid_paths) > target_frames:
                # Uniformly sample down to 4 frames
                indices = torch.linspace(0, len(valid_paths)-1, target_frames).long().tolist()
                valid_paths = [valid_paths[i] for i in indices]
            
            if len(valid_paths) < 2:
                print(f"Warning: Not enough frames for {item['id']}")
                return self.__getitem__((idx + 1) % len(self))

            # 3. Construct Conversation
            conversation = [
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
                        {"type": "text", "text": "Analyze this packaging sequence. Return JSON."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["conversations"][1]["value"]}]
                }
            ]

            # 4. Process Inputs
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            
            image_inputs, video_inputs = process_vision_info(conversation)
            
            # FIX: Explicitly handle max_length logic
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding="max_length",
                max_length=cfg["max_seq_length"],
                truncation=True,
                return_tensors="pt",
            )

            return {
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "pixel_values_videos": inputs["pixel_values_videos"] if "pixel_values_videos" in inputs else None,
                "video_grid_thw": inputs["video_grid_thw"] if "video_grid_thw" in inputs else None,
                "labels": inputs["input_ids"][0] 
            }

# --- 3. CUSTOM COLLATOR ---
@dataclass
class QwenDataCollator:
    processor: AutoProcessor

    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        
        pixel_values = []
        grid_thw = []
        
        for f in features:
            if f["pixel_values_videos"] is not None:
                pixel_values.append(f["pixel_values_videos"])
                grid_thw.append(f["video_grid_thw"])
        
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        if pixel_values:
            batch["pixel_values_videos"] = torch.cat(pixel_values, dim=0)
            batch["video_grid_thw"] = torch.cat(grid_thw, dim=0)
            
        return batch

# --- 4. MAIN TRAINING ---
def train():
    print("Loading Model...")
    
    # QLoRA Loading
    bnb_config = None
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg["model_name_or_path"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Enable Gradient Checkpointing (Crucial for VRAM)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # Prepare for LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Processor
    processor = AutoProcessor.from_pretrained(cfg["model_name_or_path"], min_pixels=256*28*28, max_pixels=1280*28*28)

    # Dataset
    train_dataset = OpenPackVideoDataset(
        data_path=cfg["data_path"],
        image_folder=cfg["image_folder"],
        processor=processor
    )
    
    print(f"Training on {len(train_dataset)} samples.")

    # Trainer Arguments
    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=float(cfg["learning_rate"]),
        fp16=cfg["fp16"],
        bf16=cfg["bf16"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        dataloader_num_workers=cfg["dataloader_num_workers"],
        remove_unused_columns=False, # Important for Custom Datasets
        report_to=cfg["report_to"]
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=QwenDataCollator(processor),
    )

    print("Starting Training...")
    trainer.train()
    
    print("Saving Adapter...")
    trainer.save_model(os.path.join(cfg["output_dir"], "final_adapter"))

if __name__ == "__main__":
    train()