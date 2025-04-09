import argparse
import datetime
import json
import os
import time
import re
import gradio as gr
import requests
from PIL import ImageDraw, Image
from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg,
    violates_moderation, moderation_msg)
import hashlib
import matplotlib.pyplot as plt


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "Mineru2 Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")
 

def add_text(state, text, image, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) < 0 and image is None:
        state.skip_next = True
        return (state, "", None)
    if image is not None:
        text = '<Mineru-Image>\n' + text + ":"
        image = Image.open(image)
        text = (text, image, "Default")
        state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return state

def state_to_md(state, image):
    md = ""
    for msg in state.messages:
        role = msg[0]
        if role != "<|im_start|>user\n":
            content = msg[1] if msg[1] is not None else ""
            md += content

    if "![](\'img_url\')" in md:
        def process_boxes_and_images(text):
            results = text
            os.makedirs('temp', exist_ok=True)
            box_pattern = re.compile(
                r'<\|box_start\|>(\d+) (\d+) (\d+) (\d+)<\|box_end\|>'
                r'<\|ref_start\|>image<\|ref_end\|>'
                r'<\|md_start\|>!\[\]\(\'(.*?)\'\)<\|md_end\|>'
            )
            for match in box_pattern.finditer(text):
                x1, y1, x2, y2 = map(int, match.groups()[:4])
                image_url = match.group(5)
                img_h, img_w = image.size
                x_1, y_1, x_2, y_2 = x1/1000*img_h, y1/1000*img_w, x2/1000*img_h, y2/1000*img_w
                if x_2 < x_1:
                    x_1, x_2 = x_2, x_1
                if y_2 < y_1:
                    y_1, y_2 = y_2, y_1
                cropped_img = image.crop((x_1, y_1, x_2, y_2))
                filename = f"{x1:03d}_{y1:03d}_{x2:03d}_{y2:03d}.png"
                save_path = os.path.join('temp', filename)
                if not os.path.isfile(save_path):
                    cropped_img.save(save_path)
                raw_text = f'<|box_start|>{x1:03d} {y1:03d} {x2:03d} {y2:03d}<|box_end|><|ref_start|>image<|ref_end|><|md_start|>![](\'{image_url}\')<|md_end|>'
                new_text = f'<|box_start|>{x1:03d} {y1:03d} {x2:03d} {y2:03d}<|box_end|><|ref_start|>image<|ref_end|><|md_start|>![](/gradio_api/file={save_path})<|md_end|>'
                results = results.replace(raw_text, new_text) 

            return results

        md = process_boxes_and_images(md)

    md = re.sub(r'<\|box_start\|>.*?<\|box_end\|>', '', md)
    md = re.sub(r'<\|ref_start\|>.*?<\|ref_end\|>', '', md)
    md = re.sub(r'<\|box_start\|>.*', '', md, flags=re.DOTALL)
    md = re.sub(r'<\|ref_start\|>.*', '', md, flags=re.DOTALL)
    md = md.replace('<|im_end|>','').replace('<|md_start|>', '').replace('<|md_end|>', '\n').strip()
    return md


def state_to_text(state):
    text = ""
    for msg in state.messages:
        role = msg[0]
        if role != "<|im_start|>user\n":
            content = msg[1] if msg[1] is not None else ""
            text += content
    return text


def state_to_img(text, image):
    def draw_boxes(image, boxes, color='red', thickness=2):
        draw = ImageDraw.Draw(image)
        img_h, img_w = image.size
        for box in boxes:
            x1, y1, x2, y2 = box[0]/1000*img_h, box[1]/1000*img_w, box[2]/1000*img_h, box[3]/1000*img_w
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        return image

    def match_boxes(text):
        boxes = []
        box_pattern = re.compile(r'<\|box_start\|>(\d+) (\d+) (\d+) (\d+)<\|box_end\|>')
        for match in box_pattern.finditer(text):
            box = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
            boxes.append(box)
        return boxes

    boxes = match_boxes(text)
    image = draw_boxes(image, boxes)
    return image

def http_bot(state, image, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    # count = 0
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state_to_md(state, image), state_to_text(state), state_to_img(state_to_text(state), image))
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "qwen" in model_name.lower():
            template_name = "qwen_2"
        elif "llava" in model_name.lower():
            if 'llama-2' in model_name.lower():
                template_name = "llava_llama_2"
            elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                if 'orca' in model_name.lower():
                    template_name = "mistral_orca"
                elif 'hermes' in model_name.lower():
                    template_name = "chatml_direct"
                else:
                    template_name = "mistral_instruct"
            elif 'llava-v1.6-34b' in model_name.lower():
                template_name = "chatml_direct"
            elif "v1" in model_name.lower():
                if 'mmtag' in model_name.lower():
                    template_name = "v1_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if 'mmtag' in model_name.lower():
                    template_name = "v0_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name.lower():
            template_name = "mpt_text"
        elif "llama-2" in model_name.lower():
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state_to_md(state, image), state_to_text(state), state_to_img(state_to_text(state), image))
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 16384),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT, SeparatorStyle.QWEN_2] else state.sep2,
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger.info(f"==== request ====\n{pload}")

    pload['images'] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state_to_md(state, image), state_to_text(state), state_to_img(state_to_text(state), image))

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state_to_md(state, image), state_to_text(state), state_to_img(state_to_text(state), image))
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state_to_md(state, image), state_to_text(state), state_to_img(state_to_text(state), image))
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state_to_md(state, image), state_to_text(state), state_to_img(state_to_text(state), image))
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state_to_md(state, image), state_to_text(state), state_to_img(state_to_text(state), image))

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def to_img(file):
    if file is None:
        return None
    return Image.open(file)

latex_delimiters = [{"left": '\\(', "right": '\\)', "display": True},
                    {"left": '\\[', "right": '\\]', "display": True},]
                    
with open("/mnt/petrelfs/liuzheng/Mineru2/Mineru_VLM-Dev/llava/serve/header.html", "r") as file:
    header = file.read()

latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
]
arabic_lang = ['ar', 'fa', 'ug', 'ur']
cyrillic_lang = [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
]
devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
        'sa', 'bgc'
]
other_lang = ['ch', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']

all_lang = ['', 'auto']
all_lang.extend([*other_lang, *latin_lang, *arabic_lang, *cyrillic_lang, *devanagari_lang])

def build_demo(embed_mode, cur_dir=None, concurrency_count=10):
    with gr.Blocks() as demo:
        state = gr.State()
        gr.HTML(header)
        with gr.Row():
            with gr.Column(variant='panel', scale=5):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)
                file = gr.File(label="Please upload an image", file_types=[".pdf", ".png", ".jpeg", ".jpg"])
                with gr.Row():
                    prompt = gr.Dropdown(["Grouding with reading order", "OCR with format and image coordinates"], label="Prompt", value="Grouding with reading order")
                    language = gr.Dropdown(all_lang, label="Language", value="auto")
                with gr.Row():
                    change_bu = gr.Button("Convert")
                    clear_bu = gr.ClearButton(value="Clear")
                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=16384, value=8096, step=64, interactive=True, label="Max output tokens",)
                img_show = gr.Image(label='Image preview', interactive=False, visible=True, height=700)
                with gr.Accordion("Examples:"):
                    example_root = os.path.join(os.path.dirname(__file__), "examples")
                    gr.Examples(
                        examples=[os.path.join(example_root, _) for _ in os.listdir(example_root) if
                                  _.endswith("jpg")],
                        inputs=file
                    )

            with gr.Column(variant='panel', scale=5):
                output_file = gr.File(label="convert result", interactive=False)
                with gr.Tabs():
                    with gr.Tab("Markdown rendering"):
                        md = gr.Markdown(label="Markdown rendering", height=1100, show_copy_button=True,
                                         latex_delimiters=latex_delimiters, line_breaks=True)
                    with gr.Tab("Markdown text"):
                        md_text = gr.TextArea(lines=45, show_copy_button=True)
        file.change(fn=to_img, inputs=file, outputs=img_show)
        change_bu.click(
            add_text,
            [state, prompt, file],
            [state]
        ).then(
            http_bot,
            [state, img_show, model_selector, temperature, top_p, max_output_tokens],
            #[state, md, md_text, output_file, img_show],
            [state, md, md_text, img_show],
            concurrency_limit=concurrency_count
        )
        clear_bu.add([file, md, img_show, md_text, output_file])

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                js=get_window_url_params
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=16)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed, concurrency_count=args.concurrency_count)
    demo.queue(
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=['./','.']
    )