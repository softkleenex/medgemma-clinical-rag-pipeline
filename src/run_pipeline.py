import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

# 1. Setup & Config
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
MODEL_ID = "google/medgemma-1.5-4b-it"
DATASET_ID = "eltorio/ROCOv2-radiology"
OUTPUT_DIR = "results"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)

def main():
    print("--- MedGemma End-to-End Pipeline ---")

    # 2. Authentication
    if not HF_TOKEN:
        print("Error: HF_TOKEN not found in .env file.")
        return
    login(token=HF_TOKEN)
    print("Logged in to Hugging Face.")

    # 3. Device Detection (Mac Metal Support)
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal Performance Shaders) acceleration.")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA acceleration.")
    else:
        device = "cpu"
        print("Using CPU (Warning: Inference will be slow).")

    # 4. Load Model & Processor
    print(f"Loading model: {MODEL_ID}...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            # variant="fp16" removed
            device_map=None # Manual device handling for MPS
        ).to(device)
        print("Model loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 5. Load Data (Streaming)
    print(f"Streaming dataset: {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    
    # 6. Inference Loop
    print("Starting inference on 3 samples...")
    results_md = "# MedGemma Inference Results\n\n"
    
    iterator = iter(dataset)
    for i in range(3):
        try:
            sample = next(iterator)
            image = sample['image']
            ground_truth = sample['caption']
            
            # Save image for report
            img_filename = f"sample_{i}.png"
            img_path = os.path.join(IMAGE_DIR, img_filename)
            image.save(img_path)

            # Prepare Prompt using official format
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this medical image."}
                ]}
            ]
            prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(device)
            
            # Generate
            generated_ids = model.generate(**inputs, max_new_tokens=150)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Cleanup output (remove prompt if echoed)
            response = generated_text.replace(prompt_text, "").strip()

            print(f"Sample {i}: Generated.")

            # Append to Report
            results_md += f"## Sample {i}\n"
            results_md += f"![Image](images/{img_filename})\n\n"
            results_md += f"**Ground Truth:** {ground_truth}\n\n"
            results_md += f"**MedGemma Prediction:** {response}\n\n"
            results_md += "---\n\n"

        except Exception as e:
            print(f"Error processing sample {i}: {e}")

    # 7. Save Report
    report_path = os.path.join(OUTPUT_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write(results_md)
    
    print(f"Done! Report saved to {report_path}")

if __name__ == "__main__":
    main()
