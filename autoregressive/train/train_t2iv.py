# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT
#   nanoGPT: https://github.com/karpathy/nanoGPT
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from glob import glob
import time
import argparse
import os

import sys
current_path = os.path.abspath(os.path.dirname(__file__))
dir1 = os.path.dirname(current_path)
sys.path.append(os.path.dirname((dir1)))


# import ipdb; ipdb.
from utils.distributed import init_distributed_mode
from utils.logger import create_logger
from dataset.build import build_dataset
from dataset.t2iv import SimpleDistributedSampler
from dataset.augmentation import center_crop_arr
from autoregressive.train.train_c2i import creat_optimizer
# from autoregressive.models.gpt import GPT_models
# from tokenizer.tokenizer_image.vq_model import VQ_models

from autoregressive.models.tokenizer.vq_models import VQ_models
from autoregressive.models.tokenizer.emu3   import Emu3VisionVQImageProcessor





def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.llm_model_hub.replace("/", "##") 
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")
    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup model: language model & tokenizer
    from autoregressive.models.qwen2 import Qwen2VisionForCausalLM
    from transformers import AutoTokenizer
    model = Qwen2VisionForCausalLM.from_pretrained(args.llm_model_hub, cache_dir="./cache_dir")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_hub, cache_dir='./cache_dir') # # "Qwen/Qwen2.5-1.5B"
    model = model.eval().to(device)
    logger.info(f"model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if  args.dataset == 't2v' or args.dataset == 't2iv':
        # MODEL_HUB = args.vq_repo # "BAAI/Emu3-VisionTokenizer"
        vq_model = VQ_models[args.vq_model]
        vq_model =  vq_model.from_pretrained(args.vq_repo,  cache_dir='./cache_dir').eval().to(device) # trust_remote_code=True,

        processor = Emu3VisionVQImageProcessor.from_pretrained(args.vq_repo, cache_dir="/storage/zhubin/UniLLM/cache_dir") # trust_remote_code=True, 
        # 暂时修改
        processor.max_pixels = 256*256
        processor.min_pixels = 256*256
        processor.size = {
            "max_pixels": 256*256,
            "min_pixels": 256*256
        },
        
    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # import ipdb; ipdb.set_trace()
    dataset = build_dataset(
                            args, 
                            tokenizer= tokenizer, 
                            processor=processor, 
                            temporal_downsample_factor=vq_model.config.temporal_downsample_factor, 
                            data_repeat=1, 
                            tokenizer_max_len=args.tokenizer_max_len       
                            )
    
    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=rank,
    #     shuffle=True,
    #     seed=args.global_seed
    # )

    sampler = SimpleDistributedSampler(
        dataset, 
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        video_sampler_batchsize=args.video_sampler_batchsize, 
        image_sampler_batchsize=args.image_sampler_batchsize, 
        video_data_step_ratio=args.video_data_step_ratio,
    )

    loader = DataLoader(
        dataset,
        # batch_size=int(args.global_batch_size // dist.get_world_size()),
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=18,
    )
    logger.info(f"Dataset contains {len(dataset):,} images")

    # Prepare models for training:
    if args.llm_ckpt:
        checkpoint = torch.load(args.llm_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.llm_ckpt.split('/')[-1].split('.')[0])
        start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.llm_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        
    
    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")

    step_counter = 0
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        # for x, y, attn_mask, valid in loader:


        for samples in loader:

            step_counter += 1
            # text 
            text = []
            for text_item in samples['text']:
                text.extend(text_item)
            inputs = tokenizer(
                text, 
                return_tensors="pt",
                # padding=True,  # 使用最大长度进行填充
                padding='max_length', 
                max_length=args.tokenizer_max_len,    # 这里替换为你想要的固定长度
                truncation=True        # 如果文本超过最大长度则截断
                )
            input_ids = inputs['input_ids'] # === (1, max_length) ===
            attention_mask = inputs['attention_mask'] # === (1, max_length) ===
            # Reverse the order of an n-D tensor along given axis in dims.
            input_ids = torch.flip(input_ids,  [1])
            attention_mask  = torch.flip(attention_mask,  [1])

            # attention_mask for (text & video)
            data_type = samples['data_type'][0]
            T  = input_ids.shape[1]

            (h, w) = samples['visual_data'][0].shape[-2:]
            if data_type == 'video':
                code_len = ((h//8) * (w//8)) * (args.num_frames//4)
            elif data_type == 'image':
                code_len = ((h//8) * (w//8)) * 1
            else:
                raise ValueError("data_type must be video or image")
                
            T_new = T + code_len
            """attention_mask = torch.cat(
                        (attention_mask, torch.ones(attention_mask.shape[0], code_len)), 
                        dim=1
                )  """
            attention_mask_matrix = torch.ones(attention_mask.shape[0], 1, T_new, T_new, dtype=torch.long)
            attention_mask_matrix  = torch.tril(attention_mask_matrix).to(torch.bool)

            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
 
            # video
            import ipdb; ipdb.set_trace()
            visual_data = torch.cat(samples['visual_data'], dim=0) # (bs, n_frame//4, 4, c, h, w)
            (b, n, t, c, h ,w) = visual_data.shape # [2, 2, 4, 3, 256, 256]  or  [2, 1, 1, 3, 512, 512]
            if data_type == 'video':
                visual_data_flat = visual_data.reshape(b * n, t, c, h, w) # [16, 4, 3, 512, 512]  or  [16, 4, 3, 256, 256] 
            elif data_type == 'image':
                visual_data = visual_data[:,0]
                visual_data_flat = visual_data.reshape(b * n, c, h, w)

            visual_data_flat = visual_data_flat.to(device, non_blocking=True)
            
            # video encode
            with torch.no_grad():
                codes = vq_model.encode(visual_data_flat)
            input_vision_ids =  codes.reshape(b, -1) # torch.Size([2, 32768])  or  [2, 8192]
            print(f'{data_type} input_vision_ids.shape: {input_vision_ids.shape}')
            
            # import ipdb; ipdb.set_trace() # print(input_ids.shape, input_vision_ids.shape, attention_mask_matrix.shape)
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                output = model(input_ids=input_ids, 
                               attention_mask_matrix=attention_mask_matrix[:, :, :-1,:-1], 
                               input_vision_ids=input_vision_ids[:,:-1],
                               labels=input_vision_ids)
                loss = output['loss']

            if step_counter % 10 == 1:
                if data_type == 'video':
                    codes = torch.argmax(output['logits'], dim=1).reshape(-1, args.num_frames//4, 32, 32) #
                elif data_type == 'image':
                    codes = torch.argmax(output['logits'], dim=1).reshape(-1, 1, 32, 32) #

                with torch.no_grad():
                    recon = vq_model.decode(codes) # (4, 4, 3, 256, 256)
                # images = recon.reshape(16, 3, 256, 256)
                recon = recon.view(-1, *recon.shape[2:])
                recon_images = processor.postprocess(recon)["pixel_values"]

                save_dir = './sample_results_new'; os.makedirs(save_dir, exist_ok=True)
                for idx, im in enumerate(recon_images):
                    im.save(f"{save_dir}/step_counter_{step_counter}_{idx}.png")
 

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            # backward pass, with gradient scaling if training in fp16         
            scaler.scale(loss).backward()

            if step_counter % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if not args.no_compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, cloud_checkpoint_path)
                    logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()


            del input_ids, attention_mask, visual_data_flat, codes
            torch.cuda.empty_cache()  # 手动清理显存

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    # parser.add_argument("--t5-feat-path", type=str, required=True)
    # parser.add_argument("--short-t5-feat-path", type=str, default=None, help="short caption of t5_feat_path")
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    # parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="Emu3_VQ")
    # parser.add_argument("--vq-repo", type=str, default=None, help="ckpt path for vq model")
    # parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    # parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    
    # parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path", type=float, default=0.0, help="drop_path_rate of attention and ffn")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='t2iv')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 


    # 添加命令行参数
    parser.add_argument('--vq-model', type=str, default="Emu3_VQ", help='Name of the VQ model to use')
    parser.add_argument('--vq-repo', type=str, default="BAAI/Emu3-VisionTokenizer", help='Repository of the VQ model')
    parser.add_argument('--num-frames', type=int, default=16, help='Number of frames to use for the video input')
    parser.add_argument('--llm_model_hub', type=str, default="Qwen/Qwen2.5-1.5B", help='LLM model hub to use')
    parser.add_argument('--tokenizer_max_len', type=int, default=512, help='Maximum length of the tokenizer sequence')
    parser.add_argument("--llm-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--data_root", type=str, default='/storage/dataset')

    # gradient_accumulation_steps
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of steps to accumulate gradients over.")
    
    # 
    parser.add_argument("--save_images", action='store_true')
    parser.add_argument("--video_data_root", type=str, default='/storage/dataset')
    parser.add_argument("--video_meta_info_file", type=str, default='/storage/zhubin/liuyihang/add_aes/output/sucai_aes_1000.json')
    parser.add_argument("--image_data_root", type=str, default='/storage/zhubin/UniLLM')
    parser.add_argument("--image_meta_info_file", type=str, default='/storage/zhubin/UniLLM/tmp/image_data.json')
    # video_data_step_ratio
    parser.add_argument("--video_data_step_ratio", type=float, default=1/3, help="Number of steps to accumulate gradients over.")
    # --video_sampler_batchsize 2
    # --image_sampler_batchsize 3
    parser.add_argument("--video_sampler_batchsize", type=int, default=2, help="Number of steps to accumulate gradients over.")
    parser.add_argument("--image_sampler_batchsize", type=int, default=3, help="Number of steps to accumulate gradients over.")
    args = parser.parse_args()
    main(args)
