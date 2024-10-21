
import torch
import sys, os

from transformers import AutoTokenizer
# MODEL_HUB =   "Qwen/Qwen-7B-Chat"
def test_Qwen():
    from autoregressive.models.qwen2 import Qwen2ForCausalLM
    MODEL_HUB =   "Qwen/Qwen2.5-1.5B"
    model = Qwen2ForCausalLM.from_pretrained(MODEL_HUB, cache_dir='./cache_dir')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HUB, cache_dir='./cache_dir')
    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)

def test_VisionQwen():
    from autoregressive.models.qwen2 import Qwen2VisionForCausalLM
    MODEL_HUB =   "Qwen/Qwen2.5-1.5B"
    # import ipdb; ipdb.set_trace()
    model = Qwen2VisionForCausalLM.from_pretrained(MODEL_HUB, cache_dir="/storage/zhubin/UniLLM/cache_dir")
    model = model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HUB, cache_dir="/storage/zhubin/UniLLM/cache_dir")
    inputs = tokenizer('a dog', return_tensors="pt")

    batch_size,  video_sequence, hidden_size = 2,  256, 768
    input_ids = inputs["input_ids"].repeat(batch_size, 1) # (1, 256)
    sequence_length = input_ids.shape[1]
    input_vision_ids = torch.ones((batch_size, video_sequence), dtype=torch.long) # (1, 256)

    # position_ids = torch.arange(sequence_length+video_sequence, dtype=torch.long).unsqueeze(0) # (1, 376)
    attention_mask = torch.ones((batch_size, sequence_length+video_sequence), dtype=torch.long) # (1, 376)
    attention_mask_matrix = torch.tril(torch.ones((batch_size,1, sequence_length+video_sequence, sequence_length+video_sequence), dtype=torch.long)) # (1, 120, 256)
    
    attention_mask_matrix = attention_mask_matrix.to(torch.bool)


    training_flag = True
    if training_flag == True:
        output = model(input_ids=input_ids, 
                    #    position_ids=position_ids, 
                    # attention_mask=attention_mask, 
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
HF_DATASETS_OFFLINES=1 python test_qwen2.py

"""