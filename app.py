import openai
from openai import OpenAI
import uvicorn
import base64
import requests
import json
from fastapi import FastAPI, Request, Form,UploadFile,File
from fastapi.responses import HTMLResponse, RedirectResponse,JSONResponse,RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from pydantic_settings import BaseSettings
import io
import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment


class Settings(BaseSettings):
    OPENAI_API_KEY: str = 'OPENAI_API_KEY'
    FLASK_APP: str = 'FLASK_APP'
    FLASK_ENV: str = 'FLASK_ENV'
    
    class Config:
        env_file = '.env'

settings = Settings()
client = OpenAI(api_key=settings.OPENAI_API_KEY)
app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Function to encode the image
def encode_image(image_path:str)->base64:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_json(file_name:str)->dict:
    # Getting the base64 string
    base64_image = encode_image(file_name)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "You are an intelligent system api that reads the texts from an image and outputs the texts as key values pairs in JSON format. Given an image, output the texts in JSON format."
                },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                }
            ]
        }
    ],
        "max_tokens": 1000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        # Save the response to a file
        with open("response_data.json", "w") as file:
            file.write(response.text)
    return response.json()
    # return templates.TemplateResponse("json_out.html", {"request": request, "data": response.json()})


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("upload_image.html", {"request": request})


@app.post("/output_iqa")
async def create_upload_file(
    request:Request,
    file: UploadFile = File(...),
    choice: str = Form(...), 
)->dict:
    pairs = {
        'quality': ("Good photo", "Bad photo"),
        'sharpness': ("Sharp photo", "Blurry photo"),
        'noisiness': ("Clean photo", "Noisy photo"),
    }
    classes = pairs[choice.lower()]
    
    _ = torch.manual_seed(42)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts the image to a PyTorch tensor
        # transforms.Normalize(mean=0,std=255)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize for pre-trained models
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Read the image file into a PIL Image
    image_content = await file.read()
    image = Image.open(io.BytesIO(image_content)).convert('RGB')
    img = transform(image)
    metric = CLIPImageQualityAssessment(
        # model_name_or_path='openai/clip-vit-large-patch14',
        prompts=(classes, choice)
    )
    out = {
        'filename':file.filename,
        'on':choice,
        "prompts":classes,
        "score": metric(img)[choice].item(),
    }
    if out['score'] > 0.75:
        # return RedirectResponse(url=f'/get_json/?filename={file.filename}')
        info = get_json(file.filename)
        out.update(json.loads(info['choices'][0]['message']['content']))
        return out
    return out


if __name__ == "__main__":
    import logging

    # Disable uvicorn access logger
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.disabled = True

    logger = logging.getLogger("uvicorn")
    logger.setLevel(logging.getLevelName(logging.DEBUG))
    uvicorn.run('app:app', host="localhost", port=5001, reload=True)
