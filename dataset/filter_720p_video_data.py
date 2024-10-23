import os, json
from tqdm import tqdm 

video_file = '/storage/zhubin/AAA_filter_crop_720P/vid_nocn_res160_json/sucai_final_3880570.json'

def read_json(video_file):
    with open(video_file, 'r') as f:
        data = json.load(f)
    return data

new_data = []
if __name__ == '__main__':
    data = read_json(video_file)
    # print(data)
    for item in tqdm(data):
        if item['resolution'] == {
                                    "height": 720,
                                    "width": 1280
                                    }:
            new_data.append(item)
    with open(f'/storage/zhubin/UniLLM/dataset/sucai_final_720p_{len(new_data)}.json', 'w') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)