import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    disable_torch_init()
    
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    
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

    image_paths = get_chunk(image_paths, args.num_chunks, args.chunk_idx)
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
        
        image = Image.open(image_path)
        image_tensor = process_images([image], image_processor, model.config)[0]
        images = image_tensor.unsqueeze(0).half().cuda()
        image_sizes = [image.size]

        if args.conv_mode == 'plain':
            prompt = DEFAULT_IMAGE_TOKEN + '\n'
        else:
            question = DEFAULT_IMAGE_TOKEN + '\nOCR with format: '
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                no_repeat_ngram_size = 20,
                length_penalty = 1,
                max_new_tokens=2048,
                do_sample=False, 
                top_k=5, 
                top_p=0.85, 
                temperature=0,
                num_return_sequences=1, 
                repetition_penalty=1.05,
                use_cache=True,
            )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a multimodal model on a set of images.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--model-base", type=str, default=None, help="Base model name or path.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing images to process.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output markdown files.")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--conv-mode", type=str, default="plain")
    
    args = parser.parse_args()
    
    eval_model(args)