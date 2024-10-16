

import os, json
from tqdm import tqdm 
DATA_FILE='/storage/zhubin/video_statistics_data/task1.5/Final_format_dataset_data_v2/step1.5_storyblocks_final_1270947_filter_1031888.json'
def read_json(file_path):
 
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

data = read_json(DATA_FILE)
new_data = []
for item in tqdm(data):
    path = '/storage/dataset/'+item['path']
    if os.path.exists(path):
        new_data.append(item)
        print(len(new_data))

with open(f'/storage/zhubin/video_statistics_data/task1.5/Final_format_dataset_data_v2/step1.5_storyblocks_final_1270947_filter_{len(new_data)}.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

# 读取 JSONL 文件