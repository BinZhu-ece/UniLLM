import os, json, glob
from tqdm import tqdm 

image_files  =  glob.glob('/storage/anno_pkl/img_nocn_res160_json/recap_64part_filter_aes_res160/**json')

def read_json(video_file):
    with open(video_file, 'r') as f:
        data = json.load(f)
    return data

new_data = []
if __name__ == '__main__':

    new_data = []
    for video_file in image_files:
        data = read_json(video_file)
 
        for item in tqdm(data):
            if item['resolution']["height"]>512 and item['resolution']["width"]>512:        
                new_data.append(item)
        print(f'new_data:{len(new_data)}!')
        if len(new_data) > 12000000:
            break
    with open(f'/storage/zhubin/UniLLM/dataset/recap_final_512+_{len(new_data)}.json', 'w') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)