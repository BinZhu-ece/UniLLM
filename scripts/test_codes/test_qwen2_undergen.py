
import torch
import sys, os
import sys
current_path = os.path.abspath(__file__)
dir1 = os.path.dirname(current_path)
dir2 = os.path.dirname(dir1)
dir3 = os.path.dirname(dir2)
print(dir3)
sys.path.append(dir3)
from transformers import AutoTokenizer
from autoregressive.models.qwen2_vl import Qwen2VLProcessor, Qwen2VLForConditionalGeneration


from autoregressive.models.qwen2_vl.qwen_vl_utils import  process_vision_info
# pip install qwen-vl-utils
from qwen_vl_utils import process_vision_info

def test_VisionQwen():
    from autoregressive.models.qwen2_vl.modeling_qwen2_vl  import Qwen2UnderGen
    # MODEL_HUB =   "Qwen/Qwen2.5-1.5B"
    MODEL_HUB = "Qwen/Qwen2-VL-2B-Instruct"
    # import ipdb; ipdb.set_trace()
    model = Qwen2UnderGen.from_pretrained(MODEL_HUB, cache_dir="/storage/zhubin/UniLLM/cache_dir")
    model = model.eval()
    
    processor = Qwen2VLProcessor.from_pretrained(MODEL_HUB,  cache_dir="/storage/zhubin/UniLLM/cache_dir")

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
    processor.image_processor.min_pixels = 128*128
    processor.image_processor.max_pixels = 256*256
    processor.image_processor.size = {
            "max_pixels": 256*256,
            "min_pixels": 128*128
        }
        
    text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # new_special_tokens = {"additional_special_tokens": ["<|enable_vision_gen_fun|>"]}
    # processor.tokenizer.add_special_tokens(new_special_tokens)
    # model.resize_token_embeddings(len(tokenizer))
    image_inputs, video_inputs = process_vision_info(messages)

    
    training_flag = True

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    batch_size,  video_sequence  = 1,  256 
    input_ids = inputs['input_ids']
    sequence_length = input_ids.shape[1]
    input_vision_ids = torch.ones((batch_size, video_sequence), dtype=torch.long) # (1, 256)

    # position_ids = torch.arange(sequence_length+video_sequence, dtype=torch.long).unsqueeze(0) # (1, 376)
    attention_mask = torch.ones((batch_size, sequence_length+video_sequence), dtype=torch.long) # (1, 376)
    attention_mask_matrix = torch.tril(torch.ones((batch_size,1, sequence_length+video_sequence, sequence_length+video_sequence), dtype=torch.long)) # (1, 120, 256)
    
    attention_mask_matrix = attention_mask_matrix.to(torch.bool)

    
    if training_flag == True:
        import ipdb; ipdb.set_trace() 

        # understanding 
        # output = model(input_ids=inputs['input_ids'], 
        #             # attention_mask_matrix=attention_mask_matrix[:, :, :-1,:-1],
        #             # input_vision_ids=input_vision_ids[:, :-1],
        #             labels = inputs['input_ids'][:, 1:]
        #             )
        # generation 
        output = model(input_ids=inputs['input_ids'], 
                    attention_mask_matrix=attention_mask_matrix[:, :, :-1,:-1],
                    input_vision_ids=input_vision_ids[:, :-1],
                    labels = input_vision_ids
                    )
    else:
        # import ipdb; ipdb.set_trace()
        output = model(input_ids=input_ids, 
                    #    position_ids=position_ids, 
                    attention_mask_matrix=attention_mask_matrix,
                    input_vision_ids=input_vision_ids,
                    # output_hidden_states=True,
                    # labels = input_vision_ids
                    )
        
        # output =  model.generate(input_ids, 
        #                 # attention_mask=attention_mask, 
        #                 attention_mask_matrix=attention_mask_matrix,
        #                 max_length=30)
        # logits  (n, 120+256, codebook_size, )
        print(output.keys())


 
    

test_VisionQwen()

"""

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
cd /storage/zhubin/UniLLM/ 
HF_DATASETS_OFFLINES=1 CUDA_VISIBLE_DEVICES=7 python /storage/zhubin/UniLLM/scripts/test_codes/test_qwen2_undergen.py

"""