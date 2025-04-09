import json
import os

data_paths = [
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage1/Vary_PDF_CN-300k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage1/Vary_PDF_EN-300k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage1/DocumentOCR-CN-1000k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage1/DocumentOCR-EN-1000k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage1/DocumentOCR-CN-Paragraph-500k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage1/DocumentOCR-EN-Paragraph-500k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage1/SceneOCR-Laion2b-IOU-Filtered-Full-37k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage1/SceneOCR-WuKong-IOU-Filtered-Full-39k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage1/SceneOCR-Laion2b-IOU-Filtered-Crop-69k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage1/SceneOCR-WuKong-IOU-Filtered-Crop-110k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage1/SceneOCR-Laion2b-IOU-Full-600k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage1/SceneOCR-WuKong-IOU-Full-600k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage1/SceneOCR-Laion2b-IOU-Crop-500k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage1/SceneOCR-WuKong-IOU-Crop-500k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage2/Mineru_cn-2000k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage2/Mineru_en-2000k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage2/Mineru_exam_zh-140k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage2/Mineru_exam_en-8k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage2/Mineru_table_zh-440k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage2/Mineru_table_en-100k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage2/Mineru_3col_zh-30k.json",
    "/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage2/Mineru_3col_en-20k.json"
]

for data_path in data_paths:
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"{data_path} has {len(data)} samples.")
    if "Stage1" in data_path:
        for item in data:
            item["conversations"][0]['value'] = "<Mineru-Image>\nOCR: "
    if "Stage2" in data_path:
        for item in data:
            item["conversations"][0]['value'] = "<Mineru-Image>\nMarkdown: "
    new_data_path = data_path.replace("Stage1", "Stage0").replace("Stage2", "Stage0")
    with open(new_data_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)