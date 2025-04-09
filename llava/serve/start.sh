proxy_off

python -m llava.serve.controller --host 10.140.24.43 --port 10000

python -m llava.serve.gradio_web_server --controller http://10.140.24.43:10000 --model-list-mode reload --host 10.140.24.43

python -m llava.serve.model_worker --host 10.140.24.43 --controller http://10.140.24.43:10000 --port 10010 --worker http://10.140.24.43:10010 --model-path /mnt/hwfile/opendatalab/liuzheng/Mineru/models/llava-ov/qwen2_bounding_box