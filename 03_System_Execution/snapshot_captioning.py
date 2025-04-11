import json
import requests
import utilities
from pathlib import Path

SNAPSHOT_CAPTIONING_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
SNAPSHOT_CAPTIONING_PROMPT = "In a short sentence, please describe the image to a blind person."
SNAPSHOT_CAPTIONING_PROMPT_HE = "במשפט קצר, תאר את התמונה לאדם עיוור"
SNAPSHOT_CAPTIONING_INTERVAL = 60  # seconds

def get_api_key():
    api_key_path = str((Path(__file__).parent.parent / Path('auxiliary/config_secret.json')).resolve().absolute())
    print(f"Importing groq API key from: {api_key_path}")
    if not Path(api_key_path).exists():
        raise FileNotFoundError(f'API key not found.\nNOTE: THE API KEY IS PRIVATE PER USER, IF YOU\'VE CLONED THIS PROJECT YOU MUST GET YOUR OWN KEY FROM: console.groq.com/keys')
    with open(api_key_path) as f:
        key = json.load(f)['GROQ_API_KEY']
    return key


def query_groq_with_image(base64_image, api_key, language='en'):
   # The image to be sent to the model (base64 format, received in function's arguments)
    image_content = {
        "type": "image_url", 
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    }
    
    # Data of the request, includes the model selected for the task and the prompt given to the model
    prompt = SNAPSHOT_CAPTIONING_PROMPT_HE if language == 'he' else SNAPSHOT_CAPTIONING_PROMPT
    data = {
        "model": SNAPSHOT_CAPTIONING_MODEL,
        "messages": [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    image_content,
                ]
            }
        ]
    }

    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_HEADERS = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.post(GROQ_URL, headers=GROQ_HEADERS, json=data)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"GROQ_ERROR: {response.status_code}: {response.text}")
    except Exception as e:
        print("Exception:", e) 
    
    return ""


#if __name__ == "__main__":
    # Potential code here to show groq request in action (without sending an image?)
    