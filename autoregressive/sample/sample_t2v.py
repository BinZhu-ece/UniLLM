import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image

import os
import time
import argparse
import sys
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
dir1 = os.path.dirname(current_directory)
dir2 = os.path.dirname(dir1)
sys.path.append(dir2)

# import ipdb; ipdb.set_trace()
from autoregressive.models.generate_t2v import generate
from autoregressive.models.tokenizer.emu3   import Emu3VisionVQImageProcessor
from autoregressive.models.tokenizer.vq_models import VQ_models

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # MODEL_HUB = args.vq_repo # "BAAI/Emu3-VisionTokenizer"
    vq_model = VQ_models[args.vq_model]
    vq_model =  vq_model.from_pretrained(args.vq_repo,  cache_dir="/storage/zhubin/UniLLM/cache_dir").eval().to(device) # trust_remote_code=True,

    processor = Emu3VisionVQImageProcessor.from_pretrained(args.vq_repo, cache_dir="/storage/zhubin/UniLLM/cache_dir") # trust_remote_code=True, 
    # 暂时修改
    processor.max_pixels = 512*512
    processor.min_pixels = 256*256
    processor.size = {
        "max_pixels": 512*512,
        "min_pixels": 256*256
    }
    
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    from autoregressive.models.qwen2 import Qwen2VisionForCausalLM
    from transformers import AutoTokenizer, AutoConfig, AutoModel


    # 加载模型的配置文件
    # config = AutoConfig.from_pretrained(args.llm_model_hub)
    # 基于配置文件初始化模型架构，但不加载预训练权重

    
    # model = AutoModel.from_config(config) # AutoModel.from_config(config)
    model = Qwen2VisionForCausalLM.from_pretrained(args.llm_model_hub, cache_dir="./cache_dir")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_hub, cache_dir='/storage/zhubin/UniLLM/cache_dir') # # "Qwen/Qwen2.5-1.5B"
    model = model.eval().to(device)

    checkpoint = torch.load(args.llm_ckpt, map_location="cpu")
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    
    gpt_model = model
    gpt_model.load_state_dict(model_weight, strict=True)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")


    # if args.compile:
    #     print(f"compiling the model...")
    #     gpt_model = torch.compile(
    #         gpt_model,
    #         mode="reduce-overhead",
    #         fullgraph=True
    #     ) # requires PyTorch 2.0 (optional)
    # else:
    #     print(f"no need to compile model in demo") 
    
    # prompt = [
    #     "A dog is running.",
    # ]
    prompt = [
            # "The video showcases a person using a red tablet, which is mounted on a steering wheel, to interact with what appears to be a vehicle's infotainment system or diagnostic tool. Initially, the tablet displays a menu with various options, including \"Basic Settings\" and \"Live View Camera,\" highlighting the system's functionality within a vehicle context. The person's right hand is seen selecting or navigating through these options. As the video progresses, the screen changes to display a graph with a blue waveform, likely representing vehicle performance metrics such as engine RPM or tire pressure, indicating that the user has navigated to a performance monitoring or diagnostic menu. The person continues to interact with the tablet, adjusting settings or scrolling through the diagnostic information. Throughout these interactions, the focus remains on the tablet and the user's hands, with the background consistently blurred, emphasizing the detailed operation of the vehicle's infotainment system and the user's engagement with the device.",
            "In the video, a person's hand is seen interacting with a tablet device, which is displaying a graphical user interface with a variety of numerical data and graphs. The tablet screen is filled with information, suggesting that the user might be monitoring or analyzing data in real-time or reviewing historical data. The graphs and numerical data indicate that the content is technical or scientific in nature, possibly related to engineering, physics, or another field that requires detailed analysis. The user's fingers are actively engaging with the touchscreen, possibly scrolling, tapping, or swiping to navigate through the application or to manipulate the data displayed. The overall scene conveys a sense of focus and engagement with the content on the tablet."
        ]

    MODEL_HUB =   "Qwen/Qwen2.5-1.5B"
    # caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HUB, cache_dir='/storage/zhubin/UniLLM/cache_dir')
    # prompt = "Hey, are you conscious? Can you talk to me?"
    # inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True)
    input_ids  = inputs['input_ids'].to(device)
    # input_ids  = inputs['input_ids'] 

    # qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
    
    t1 = time.time()
    index_sample = generate(
        gpt_model, input_ids,  
        # max_new_tokens=(256//8)**2*16//4, 
        max_new_tokens=(256//8)**2*4//4, 
        cfg_scale=args.cfg_scale,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True, 
        ) # (1, 4*1*32*32)
    sampling_time = time.time() - t1
    print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
    
    import ipdb; ipdb.set_trace()
    codes = index_sample.reshape(-1, 1, 32, 32) # (4, 1, 32, 32)
 
    with torch.no_grad():
        recon = vq_model.decode(codes) # (4, 4, 3, 256, 256)
    # images = recon.reshape(16, 3, 256, 256)
    recon = recon.view(-1, *recon.shape[2:])
    recon_images = processor.postprocess(recon)["pixel_values"]
    for idx, im in enumerate(recon_images):
        im.save(f"recon_video_new_{idx}.png")
    t2 = time.time()
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    # save_image(recon, "sample_{}.png".format(args.gpt_type), nrow=4, normalize=True, value_range=(-1, 1))
    # print(f"image is saved to sample_{args.gpt_type}.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    # parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    # parser.add_argument("--t5-feature-max-len", type=int, default=120)
    # parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    # parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    # parser.add_argument("--gpt-ckpt", type=str, default=None)
    # parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    # parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    # parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")

    # 添加命令行参数
    parser.add_argument('--vq-model', type=str, default="Emu3_VQ", help='Name of the VQ model to use')
    parser.add_argument('--vq-repo', type=str, default="BAAI/Emu3-VisionTokenizer", help='Repository of the VQ model')
    parser.add_argument('--num-frames', type=int, default=16, help='Number of frames to use for the video input')
    parser.add_argument('--llm_model_hub', type=str, default="Qwen/Qwen2.5-1.5B", help='LLM model hub to use')
    parser.add_argument('--tokenizer_max_len', type=int, default=512, help='Maximum length of the tokenizer sequence')
    parser.add_argument("--llm-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--video_meta_info_file", type=str, default='/storage/zhubin/liuyihang/add_aes/output/sucai_aes_1000.json')
    parser.add_argument("--data_root", type=str, default='/storage/dataset')
    args = parser.parse_args()
    main(args)



    """
    cd  /storage/zhubin/UniLLM
    DATA_FILE='/storage/zhubin/video_statistics_data/task1.5/Final_format_dataset_data_v2/step1.5_storyblocks_final_1270947_filter_1031888.json'

    nnodes=1
    nproc_per_node=1
    export master_addr=127.0.0.1
    export master_port=29509
    export CUDA_VISIBLE_DEVICES=1


    source  /storage/miniconda3/etc/profile.d/conda.sh 
    conda activate 

    torchrun \
    --nnodes=$nnodes --nproc_per_node=$nproc_per_node  \
    --master_addr=$master_addr --master_port=$master_port \
    autoregressive/sample/sample_t2v.py \
    --num-frames 16 \
    --llm_model_hub Qwen/Qwen2.5-1.5B \
    --tokenizer_max_len 512 \
    --vq-model   Emu3_VQ \
    --vq-repo  BAAI/Emu3-VisionTokenizer \
    --llm-ckpt  /storage/zhubin/UniLLM/results/1.5B-16f-256px/007-Qwen##Qwen2.5-1.5B/checkpoints/0035000.pt

 
    
    
    """