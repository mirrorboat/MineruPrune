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
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
print("hello")


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        instructs = self.tokenizer(qs.replace(DEFAULT_IMAGE_TOKEN, '').replace(DEFAULT_IM_START_TOKEN, '').replace(DEFAULT_IM_END_TOKEN, '').strip()).input_ids
        instructs = torch.tensor(instructs, dtype=torch.long)
        return input_ids, image_tensor, image.size, instructs

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, instructs = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    instructs = torch.stack(instructs, dim=0)
    return input_ids, image_tensors, image_sizes, instructs


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

from transformers import AutoTokenizer


def batch_cosine_similarity(img_tokens, embedding_matrix, batch_size=10):
    num_img_tokens = img_tokens.size(0)
    all_cos_similarities = []

    for i in range(0, num_img_tokens, batch_size):
        batch_img_tokens = img_tokens[i:i+batch_size]
        # 计算当前批次的余弦相似度
        cos_similarities_batch = torch.nn.functional.cosine_similarity(batch_img_tokens, embedding_matrix.unsqueeze(0), dim=-1)
        # 改为计算内积
        # cos_similarities_batch = torch.matmul(batch_img_tokens, embedding_matrix.t())
        all_cos_similarities.append(cos_similarities_batch)

    # 将所有批次的结果拼接在一起
    all_cos_similarities = torch.cat(all_cos_similarities, dim=0)
    return all_cos_similarities


SIZE=576

image_token_ls=[]
image_token_ls1=[]
image_token_ls2=[]
height=weight=int(math.sqrt(SIZE))
def image_hook(module, input, output):
    # image_token_ls.append(output)
    output_copy=output.clone().detach()
    image_token_ls.append(output_copy)

    # if count % 2 == 0:
    #     # image_token_ls1.append(output_copy[...,:SIZE,:]).reshape(image_features.shape[0], height, width, -1)
    #     image_token_ls1.append(output_copy[...,:SIZE,:])
    # else:
    #     # image_token_ls2.append(output_copy.view(output_copy.shape[0], height, weight, -1).permute(0, 3, 1, 2))
    #     image_token_ls2.append(output_copy)

height = width = int(math.sqrt(SIZE))
# image_features = image_features.reshape(image_features.shape[0], height, width, -1)

def eval_model(args):
    # Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    embedding_matrix = model.get_model().embed_tokens.weight.detach()
    print(f"size of embedding matrix: {embedding_matrix.size()}")
    # model.encoder.visual_encoder.register_forward_hook(image_hook)
    model.get_model().mm_projector.register_forward_hook(image_hook)
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes, instructs), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        instructs = instructs.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                instructs=instructs)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        # ans_file.write(json.dumps({"question_id": idx,
        #                            "prompt": cur_prompt,
        #                            "text": outputs,
        #                            "answer_id": ans_id,
        #                            "model_id": model_name,
        #                            "metadata": {}}) + "\n")
        # ans_file.flush()
    # ans_file.close()
    

        # image_token_ls的偶数位置作为image_token_ls1，奇数位置作为image_token_ls2
        image_token_ls1 = image_token_ls[::2]
        image_token_ls1 = [feature[...,:SIZE,:] for feature in image_token_ls1]
        image_token_ls2 = image_token_ls[1::2] # [1, 2*2*SIZE, dim]
        before_allocated = torch.cuda.memory_allocated()
        before_reserved = torch.cuda.memory_reserved()
        del model
        torch.cuda.empty_cache()
        # 打印显存情况
        # torch.cuda.memory_summary(device=None, abbreviated=False)
        torch.cuda.synchronize() 
        torch.cuda.ipc_collect()
        # 记录清理后的显存使用情况
        after_allocated = torch.cuda.memory_allocated()
        after_reserved = torch.cuda.memory_reserved()
        # 打印对比
        print(f"Allocated memory before: {before_allocated / 1024 ** 2:.2f} MB")
        print(f"Allocated memory after:  {after_allocated / 1024 ** 2:.2f} MB")
        print(f"Reserved memory before:  {before_reserved / 1024 ** 2:.2f} MB")
        print(f"Reserved memory after:   {after_reserved / 1024 ** 2:.2f} MB")
        
        top_count=10
        for id, img_token in enumerate(image_token_ls1):
            img_token = img_token.squeeze(0).unsqueeze(1) # [SIZE, 1, dim]
            # TODO: 将img_token切块后再计算cosine similarity，避免OOM
            cos_similarities = batch_cosine_similarity(img_token, embedding_matrix, batch_size=10) # [SIZE, vocab_size]
            # 将cos_similarities中所有nan替换为-100
            cos_similarities[torch.isnan(cos_similarities)] = -100 # torch.topk会优先取出nan
            # 获取cos_similarities每个位置top10值及其索引，也就是得到两个形状为[SIZE, top_count]的张量
            topk_values, topk_indices = torch.topk(cos_similarities, k=top_count, dim=-1)
            # 将topk_values,topk_indices变形为[height, width, top_count]
            topk_values = topk_values.view(height, width, top_count)
            topk_indices = topk_indices.view(height, width, top_count)
            # 将topk_indices转换为对应的token，存入形状为[height, width, top_count]的三维列表
            topk_tokens = []
            for i in range(height):
                row = []
                for j in range(width):
                    cell=[]
                    for k in range(top_count):
                        token = tokenizer.decode(topk_indices[i,j,k].item())
                        cell.append(token)
                    row.append(cell)
                topk_tokens.append(row)
            # 将topk_tokens写入txt文件，每个位置的top10 token占一行
            with open(f"topk_tokens_{id}.txt", "w") as f:
                for i in range(height):
                    for j in range(width):
                        f.write(" ".join(topk_tokens[i][j]) + "\n")
                    f.write("\n")
            breakpoint()


        for img_token in image_token_ls2:
            img_token = img_token.squeeze(0).unsqueeze(1) # [4*SIZE, 1, dim]
            cos_similarities = batch_cosine_similarity(img_token, embedding_matrix, batch_size=10) # [4*SIZE, vocab_size]
            topk_values, topk_indices = torch.topk(cos_similarities, k=top_count, dim=-1)
            topk_values = topk_values.view(2*height, 2*width, top_count)
            topk_indices = topk_indices.view(2*height, 2*width, top_count)

if __name__ == "__main__":
    print("hello")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/petrelfs/chenjingzhou/cjz/HINT/playground/training/training_dirs/HINT-8B-Finetune")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/mnt/petrelfs/chenjingzhou/cjz/HINT/playground/eval/pope/val2014")
    parser.add_argument("--question-file", type=str, default="/mnt/petrelfs/chenjingzhou/cjz/HINT/eval.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
