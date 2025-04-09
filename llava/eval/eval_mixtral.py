import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
import base64
import markdown
import requests
import os
from mistralai import Mistral
import json
from io import BytesIO
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

api_key = "zdoj0CXuYD1JXWJTKvNAicpC0YOnLaak"
client = Mistral(api_key=api_key)

def eval_model(args):
    output_path = os.path.expanduser(args.output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    data_path = os.path.expanduser(args.data_dir)
    
    image_files = os.listdir(data_path)
    image_paths = []
    has_subdirectories = False
    
    for image_file in image_files:
        full_path = os.path.join(data_path, image_file)
        if image_file.lower().endswith('.json'):
            continue
        if os.path.isdir(full_path):
            has_subdirectories = True
            sub_images = os.listdir(full_path)
            for sub_image in sub_images:
                sub_full_path = os.path.join(full_path, sub_image)
                if sub_full_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_paths.append(sub_full_path)
        else:
            if full_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(full_path)

    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing Images"):
        if has_subdirectories:
            dir_name = os.path.basename(os.path.dirname(image_path))
            base_name = os.path.basename(image_path)
            file_name = f"{dir_name}-{base_name.rsplit('.', 1)[0]}.md"
        else:
            base_name = os.path.basename(image_path)
            file_name = f"{base_name.rsplit('.', 1)[0]}.md"
        save_path = os.path.join(output_path, file_name)
        if os.path.exists(save_path):
            continue
        try:
            base64_image = encode_image(image_path)
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}" 
                }
                ,
                include_image_base64=True
            )
            response_dict = json.loads(ocr_response.json())
            json_string = json.dumps(response_dict, indent=4)
            markdown_string = ocr_response.pages[0].markdown
                
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(markdown_string)
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a multimodal model on a set of images.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--model-base", type=str, default=None, help="Base model name or path.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing images to process.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output markdown files.")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    
    args = parser.parse_args()
    
    eval_model(args)