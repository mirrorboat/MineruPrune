data=[
"/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_en800k_box-type-md.json",
"/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_zh800k_box-type-md.json",
"/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_exam_en_box-type-md.json",
"/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_exam_zh_box-type-md.json",
"/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_table_en_box-type-md.json",
"/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_table_zh_box-type-md.json",
"/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_3col_en_box-type-md.json",
"/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_3col_zh_box-type-md.json",
"/mnt/hwfile/opendatalab/bigdata_mineru/liuzheng/Mineru/dataset/bounding_box/Mineru_Notes_box-type-md.json",
"/mnt/hwfile/opendatalab/bigdata_mineru/liuzheng/Mineru/dataset/bounding_box/Mineru_Notes_Math_box-type-md.json",
# "/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_newspaper420k_box-type-md.json",
]

# data=[
# "/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_exam_zh_box-type-md.json",
# "/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_table_zh_box-type-md.json",
# "/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_3col_zh_box-type-md.json",
# "/mnt/hwfile/opendatalab/bigdata_mineru/liuzheng/Mineru/dataset/bounding_box/Mineru_Notes_box-type-md.json",
# "/mnt/hwfile/opendatalab/bigdata_mineru/liuzheng/Mineru/dataset/bounding_box/Mineru_Notes_Math_box-type-md.json",
# "/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_newspaper420k_box-type-md.json",
# ]

# data=[
# "/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_en800k_box-type-md.json",
# "/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_exam_en_box-type-md.json",
# "/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_table_en_box-type-md.json",
# "/mnt/hwfile/opendatalab/guzhuangcheng/data/mineru_extract/grounding/got-mmd/filtered2/mineru2_inhouse_3col_en_box-type-md.json",
# ]

# 从每个json文件中随机抽取25条数据，合并为新的json文件
import json
import random
import os
import shutil

num_samples=50
selected_data = []
for file_path in data:
    with open(file_path, 'r') as f:
        file_data = json.load(f)
        # print(f"File {file_path} contains {len(file_data)} samples.")
        # 随机抽取num_samples条数据
        if len(file_data) > num_samples:
            selected_data.extend(random.sample(file_data, num_samples))
        else:
            selected_data.extend(file_data)

# shuffle
random.shuffle(selected_data)

# 保存到新的json文件
output_file = "selected_subset_all.json"
with open(output_file, 'w') as f:
    json.dump(selected_data, f, indent=4)

print(f"Selected {len(selected_data)} samples and saved to {output_file}")