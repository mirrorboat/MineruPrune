import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import pdb
import gradio as gr
import requests

from llava.constants import LOGDIR
from llava.conversation import SeparatorStyle, conv_templates, default_conversation
from llava.utils import (
    build_logger,
    moderation_msg,
    server_error_msg,
    violates_moderation,
)

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "LLaVA Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

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

    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()

    models_downloaded = True if models else False

    model_dropdown_kwargs = {
        "choices": [],
        "value": "Downloading the models...",
        "interactive": models_downloaded,
    }

    if models_downloaded:
        model_dropdown_kwargs["choices"] = models
        model_dropdown_kwargs["value"] = models[0]

    models_dropdown_update = gr.Dropdown.update(**model_dropdown_kwargs)

    send_button_update = gr.Button.update(
        interactive=models_downloaded,
    )

    return state, models_dropdown_update, send_button_update


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


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def add_text(state, text, image, image_process_mode, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,
            ) * 5

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if "<image>" not in text:
            # text = '<Image><image></Image>' + text
            text = text + "\n<image>"
        text = (text, image, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def http_bot(
    state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request
):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "llava" in model_name.lower():
            if "llama-2" in model_name.lower():
                template_name = "llava_llama_2"
            elif "v1" in model_name.lower():
                if "mmtag" in model_name.lower():
                    template_name = "v1_mmtag"
                elif (
                    "plain" in model_name.lower()
                    and "finetune" not in model_name.lower()
                ):
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if "mmtag" in model_name.lower():
                    template_name = "v0_mmtag"
                elif (
                    "plain" in model_name.lower()
                    and "finetune" not in model_name.lower()
                ):
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (
            state,
            state.to_gradio_chatbot(),
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(
            LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg"
        )
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep
        if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT]
        else state.sep2,
        "images": f"List of {len(state.get_images())} images: {all_image_hash}",
    }
    logger.info(f"==== request ====\n{pload}")

    pload["images"] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=pload,
            stream=True,
            timeout=10,
        )
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt) :].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


title_markdown = """
# 🌋 LLaVA: Large Language and Vision Assistant
[[Project Page]](https://llava-vl.github.io) [[Paper]](https://arxiv.org/abs/2304.08485) [[Code]](https://github.com/haotian-liu/LLaVA) [[Model]](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)

ONLY WORKS WITH GPU!

You can load the model with 4-bit or 8-bit quantization to make it fit in smaller hardwares. Setting the environment variable `bits` to control the quantization.
*Note: 8-bit seems to be slower than both 4-bit/16-bit. Although it has enough VRAM to support 8-bit, until we figure out the inference speed issue, we recommend 4-bit for A10G for the best efficiency.*

Recommended configurations:
| Hardware          | T4-Small (16G)  | A10G-Small (24G) | A100-Large (40G) |
|-------------------|-----------------|------------------|------------------|
| **Bits**          | 4 (default)     | 4                | 16               |

"""

tos_markdown = """
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
"""


learn_more_markdown = """
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
"""

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""


def build_demo(embed_mode):
    models = get_model_list()

    textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", container=False
    )
    with gr.Blocks(title="LLaVA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State(default_conversation.copy())

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if models else "Downloading the models...",
                        interactive=True if models else False,
                        show_label=False,
                        container=False,
                    )

                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image",
                    visible=False,
                )

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(
                    examples=[
                        [
                            f"{cur_dir}/examples/extreme_ironing.jpg",
                            "What is unusual about this image?",
                        ],
                        [
                            f"{cur_dir}/examples/waterview.jpg",
                            "What are the things I should be cautious about when I visit here?",
                        ],
                    ],
                    inputs=[imagebox, textbox],
                )

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=1024,
                        value=512,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot", label="LLaVA Chatbot", height=550
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(
                            value="Send", variant="primary", interactive=False
                        )
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
        )
        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
        )
        clear_btn.click(
            clear_history, None, [state, chatbot, textbox, imagebox] + btn_list
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
        )
        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                _js=get_window_url_params,
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list, None, [state, model_selector, submit_btn]
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


def start_controller():
    logger.info("Starting the controller")
    controller_command = [
        "python",
        "-m",
        "llava.serve.controller",
        "--host",
        "0.0.0.0",
        "--port",
        "10000",
    ]
    return subprocess.Popen(controller_command)


def start_worker(model_path: str, bits=16):
    logger.info(f"Starting the model worker for the model {model_path}")
    model_name = model_path.strip("/").split("/")[-1]
    assert bits in [4, 8, 16], "It can be only loaded with 16-bit, 8-bit, and 4-bit."
    if bits != 16:
        model_name += f"-{bits}bit"
    worker_command = [
        "python",
        "-m",
        "llava.serve.model_worker",
        "--host",
        "0.0.0.0",
        "--controller",
        "http://localhost:10000",
        "--model-path",
        model_path,
        "--model-name",
        model_name,
    ]
    if bits != 16:
        worker_command += [f"--load-{bits}bit"]
    return subprocess.Popen(worker_command)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:10000")
    parser.add_argument("--concurrency-count", type=int, default=8)
    parser.add_argument(
        "--model-list-mode", type=str, default="reload", choices=["once", "reload"]
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")

    args = parser.parse_args()

    return args


def start_demo(args):
    demo = build_demo(args.embed)
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    args = get_args()
    logger.info(f"args: {args}")

    model_path = "/mnt/petrelfs/liuzheng/Mineru2/Mineru_VLM/playground/training/training_dirs/Qwen2-0.5B-Siglip-Pretrain-Qwen2-0.5B-Stage2-checkpoint-2"
    bits = int(os.getenv("bits", 16))

    controller_proc = start_controller()
    time.sleep(10)
    worker_proc = start_worker(model_path, bits=bits)
    time.sleep(10)

    # Wait for worker and controller to start
    time.sleep(10)

    exit_status = 0
    try:
        start_demo(args)
    except Exception as e:
        # pdb.set_trace()
        print(e)
        exit_status = 1
    finally:
        worker_proc.kill()
        controller_proc.kill()

        sys.exit(exit_status)
