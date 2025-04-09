import json
with open("/mnt/hwfile/opendatalab/liuzheng/Mineru/dataset/Stage2/Mineru_Notes_100K.json") as f:
    data = json.load(f)

print(data[0])
print(data[-1])
# for item in data:
#     assert 'image' not in item