import os
import glob
import requests
from dotenv import load_dotenv

# 1. Load Config
load_dotenv()
API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "RGb96Dcl0k5eVje8EBch" # CORRECTED Serena ID

def generate_narration():
    script_files = sorted(glob.glob("results/narration/*.txt"))
    
    if not script_files:
        print("No script files found in results/narration/")
        return

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY
    }

    print(f"Starting audio generation via Direct API for {len(script_files)} files...")

    for script_path in script_files:
        filename = os.path.basename(script_path).replace(".txt", ".mp3")
        output_path = os.path.join("results/narration", filename)
        
        with open(script_path, "r") as f:
            text = f.read()
        
        print(f"Generating audio for: {filename}...")
        
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                print(f"Successfully saved to {output_path}")
            else:
                print(f"Failed to generate {filename}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Request Error for {filename}: {e}")

if __name__ == "__main__":
    generate_narration()
