# Mineru2.0
## 环境配置：
和llava环境大致保持一致，大致步骤如下：
1. Clone this repository and navigate to LLaVA folder
```
git clone https://github.com/opendatalab/MinerU-LVLM.git
cd MinerU-LVLM
```

2. Install Package
```
conda create -n mineru python=3.10 -y
conda activate mineru
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
4. Upgrade to latest code base(Optional)
```
git pull
pip install -e .# if you see some import errors when you upgrade,# please try running the command below (without #)# pip install flash-attn --no-build-isolation --no-cache-dir
```
## 数据准备：
数据格式和LLaVA保持一致，在json文件中以列表存储，列表每一项是一个字典，示例格式如下：
```json
    {
        "id": "mineru:s3://doc-parse-huawei/mineru2/ocr-sence-en-laion2b/full/1000001003407.0.jpg",
        "image": "mineru:s3://doc-parse-huawei/mineru2/ocr-sence-en-laion2b/full/1000001003407.0.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<Mineru-Image>\n"
            },
            {
                "from": "gpt",
                "value": "THEHERD"
            }
        ]
    }
```
## 训练 & 评测
### 训练数据：
LLaVA格式的Stage1 V4数据集存放在/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset，可以直接使用。

### 训练模型：
所有的模型均采用huggingface格式，根据仓库名在huggingface中下载即可。

Sam encoder：facebook/sam-vit-base

OPT-150M：facebook/opt-150m

Qwen2-0.5B-Instruct：Qwen/Qwen2-0.5B-Instruct

### 运行脚本：
通过sh/start-opt.sh或sh/start-qwen.sh运行，支持多节点GPU和deepspeed加速。

### 评测脚本：
通过inference/run-opt.sh或inference/run-scene.sh来获取对GOT Benchmark和Scene的评测结果。支持多节点GPU并行评测。

评测的编辑距离等结果需要通过Omnidocbench获取。

## 整体仓库格式：
```
Mineru:
  ├── MinerU-LVLM
          ├── data_recipe
          ├── inference
                  ├── run-opt.sh
                  └── run-scene.sh
          ├── llava
          ├── scripts
          │    └── zero3.json
          └── sh
               ├── start-opt.sh
               └── start-qwen.sh
  ├── playground
          ├── dataset
          ├── model
               ├── opt-125m
               └── Qwen2-0.5B-Instruct
               └── sam-vit-base
          ├── training
               └── training_dirs
```