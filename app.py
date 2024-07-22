import openai
from openai import OpenAI
import uvicorn
import base64
import requests
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
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts the image to a PyTorch tensor
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize for pre-trained models
    #                     std=[0.229, 0.224, 0.225])
])

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Function to encode the image
def encode_image(image_path:str)->base64:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("upload_image.html", {"request": request})


@app.post("/get_json/{}", response_class=HTMLResponse)
def index(request: Request,file_name: str):
    # Getting the base64 string
    base64_image = encode_image(file_name)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "What’s in this image?"
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
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())
    return response.json()
    # return templates.TemplateResponse("json_out.html", {"request": request, "data": response.json()})

@app.post("/output_iqa")
async def create_upload_file(request:Request,file: UploadFile = File(...)):
    # Read the image file into a PIL Image
    print(file)
    image_content = await file.read()
    image = Image.open(io.BytesIO(image_content)).convert('RGB')
    img = transform(image).unsqueeze(0)
    metric = CLIPImageQualityAssessment(
        model_name_or_path='openai/clip-vit-base-patch16',
        prompts=(("Good Photo.", "Bad Photo."), "quality")
    )
    out = {
        'on':"quality",
        "prompts":("Good Photo.", "Bad Photo."),
        "score": metric(img)['quality'],
    }
    if out['score'] > 0.7:
        return RedirectResponse(url=f'/get_json/?filename={file.filename}')
    
    return templates.TemplateResponse(
        "json_out.html", 
        {"request": request, "data": out}
    )

    # Respond with the random float and image info
    return JSONResponse(content={"IQA": metric(img),'text':'wtf'})








if __name__ == "__main__":
    uvicorn.run('app:app', host="localhost", port=5001, reload=True)
