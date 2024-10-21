from PIL import Image
import requests, os
# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

import sys
current_path = os.path.abspath(__file__)
dir1 = os.path.dirname(current_path)
dir2 = os.path.dirname(dir1)
dir3 = os.path.dirname(dir2)
# print(dir3)
sys.path.append(dir3)

from autoregressive.models.qwen2_vl import Qwen2VLProcessor, Qwen2VLForConditionalGeneration

from autoregressive.models.qwen2_vl.qwen_vl_utils import  process_vision_info
# pip install qwen-vl-utils
from qwen_vl_utils import process_vision_info

def main():
    # 加载模型和处理器
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name,  torch_dtype="auto", cache_dir="/storage/zhubin/UniLLM/cache_dir").to("cuda")
    processor = Qwen2VLProcessor.from_pretrained(model_name,  cache_dir="/storage/zhubin/UniLLM/cache_dir")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "/storage/zhubin/UniLLM/video_samples/c2i7cCwRuw4_segment_1.mp4",
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "A dog is running."},
    #         ],
    #     }
    # ]
    # Preparation for inference
    

    processor.image_processor.min_pixels = 128*128
    processor.image_processor.max_pixels = 256*256
    processor.image_processor.size = {
            "max_pixels": 256*256,
            "min_pixels": 128*128
        }
        
    text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # import ipdb; ipdb.set_trace()
    new_special_tokens = {"additional_special_tokens": ["<|enable_vision_gen_fun|>"]}
    processor.tokenizer.add_special_tokens(new_special_tokens)
    # model.resize_token_embeddings(len(tokenizer))


    image_inputs, video_inputs = process_vision_info(messages)

    

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    
    training = True
    if training:
        output = model(**inputs)
    else:
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)

if __name__ == "__main__":
    main()


"""

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
cd /storage/zhubin/UniLLM/ 
CUDA_VISIBLE_DEVICES=6 python scripts/test_codes/qwen2-vl-generation.py
HF_DATASETS_OFFLINES=1
"""