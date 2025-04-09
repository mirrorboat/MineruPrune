import json

# data = []
# with open("/mnt/hwfile/opendatalab/zhangrui/shared_data/mineru_lvlm_data/notes_math_50K.jsonl") as f:
#     for line in f:
#         data.append(json.loads(line))

# with open("/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage2/Mineru_Notes_Math_50K.json", "w") as f:
#     json.dump(data, f, indent=4)

with open("/mnt/hwfile/opendatalab/zhangrui/shared_data/mineru2_data/mineru_newspaper_zh150K.json") as f:
    data = json.load(f)

for item in data:
    item['conversations'][0]['value'] = "<Mineru-Image>\nOCR with format: "
    item['image'] = "mineru:s3://doc-parse-huawei/mineru2/inhouse-markdown/newpaper_zh-700-K/v001/images/" + item['image']

with open("/mnt/petrelfs/liuzheng/Mineru2/playground/dataset/Stage2/Mineru_newspaper_zh-150k.json", "w") as f:
    json.dump(data, f, indent=4)


