# https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html
import torch
import json
from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment
from torchvision import transforms
from PIL import Image


def load_image(
    image_path:str,
    transforms:transforms.Compose
)->torch.tensor:
    # Open the image file
    img = Image.open(image_path).convert('RGB')  # Convert to RGB
    img_tensor = transform(img)  # Apply the transformation
    return img_tensor

def test_json()->None:
    path = "response_data.json"
    with open(path, 'r') as f:
        res = json.load(f)
    print(json.loads(res['choices'][0]['message']['content']))
    
if __name__ == '__main__':
    # path = 'Complex Imaging Tags _ Metal Marker Mfg (1).jpeg'
    # _ = torch.manual_seed(42)
    # # Define a transform to convert the image to a tensor and normalize it
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # Converts the image to a PyTorch tensor
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize for pre-trained models
    #     #                     std=[0.229, 0.224, 0.225])
    # ])
    # img = load_image(path,transform).unsqueeze(0).float()
    # print(img.shape)
    # metric = CLIPImageQualityAssessment(
    #     model_name_or_path='openai/clip-vit-base-patch16',
    #     prompts=(("Good Photo.", "Bad Photo."), "quality")
    # )
    # print(metric(img))
    test_json()