import argparse
import json
import os, io, csv, math, random
import numpy as np
import torchvision
from einops import rearrange
from decord import VideoReader
from os.path import join as opj
import gc
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from PIL import Image
import sys
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_directory, '..'))
# from dataset.utils.dataset_utils import DecordInit
# from dataset.utils.utils import text_preprocessing

from utils.distributed import init_distributed_mode
from decord import cpu, gpu, VideoReader

def random_video_noise(t, c, h, w):
    vid = torch.rand(t, c, h, w) * 255.0
    vid = vid.to(torch.uint8)
    return vid

"""
video_meta_info_file
{
    "path": "mixkit/sunset/mixkit-lightbulb-in-snow-2569_resize1080p.mp4",
    "cap": [
      "The video depicts a serene and picturesque winter sunset scene throughout its duration. It begins with a tranquil atmosphere, featuring an illuminated light bulb resting on the snowy ground in the foreground, with a warm glow emanating from it. The background showcases a twilight sky blending hues of orange and blue, with silhouettes of bare trees visible against the horizon. Throughout the video, there are no noticeable changes or movements within the scene. The light bulb remains luminous, the sunset colors persist in their blend, and the tree silhouettes retain their position against the sky. This consistent imagery conveys a sense of stillness and tranquility, maintaining the winter evening's serene ambiance from start to finish."
    ],
    "size": 4302912,
    "duration": 15.5155,
    "resolution": {
      "width": 1920,
      "height": 1080
    },
    "frames": 372,
    "fps": 23.976023976023978,
    "aspect_ratio": "16:9",
    "motion": 0.9993534684181213,
    "motion_average": 0.9988254904747009
  },
"""

from torch.utils.data import Dataset, DataLoader

from language.t5 import T5Embedder


import matplotlib.pyplot as plt
def visualize_attn_mask(attn_mask, save_path):
    # 如果 attn_mask 是 3D tensor，去掉 batch 维度
    attn_mask_2d = attn_mask.squeeze(0).cpu().numpy()  # 转换为 2D numpy 数组以便可视化

    # 创建一个 figure 和 axis
    plt.figure(figsize=(10, 10))  # 可以调整 figure 大小
    plt.imshow(attn_mask_2d, cmap='gray', interpolation='none')  # 使用灰度色彩

    # 添加标题和标签
    plt.title('Attention Mask Visualization')
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')

    # 显示 colorbar 以指示数值范围
    plt.colorbar(label='Mask Value')
    # 显示图像
    plt.savefig(save_path)

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import math

class SimpleDistributedSampler(Sampler):
    def __init__(self, data_source, num_replicas=None, rank=None, shuffle=False, 
                epoch=1, video_sampler_batchsize=2, image_sampler_batchsize=3, video_data_step_ratio=1/4,
                ):
        
        self.data_source = data_source # dataset
        # print(self.data_source)
        self.num_replicas = num_replicas if num_replicas is not None else torch.distributed.get_world_size()
        self.rank = rank if rank is not None else torch.distributed.get_rank()
        self.shuffle = shuffle
    
        # 每个进程负责的样本数
        self.num_samples = math.ceil(len(self.data_source) / self.num_replicas)
        
        # 数据总量对齐到进程数的倍数
        self.total_size = self.num_samples * self.num_replicas
        self.epoch = epoch
        self.video_data_step_ratio = video_data_step_ratio # 视频训练iteration占总iteration的比例
        self.video_sampler_batchsize = video_sampler_batchsize # getitem一次返回一个视频idx列表，列表长度为video_sampler_batchsize
        self.image_sampler_batchsize = image_sampler_batchsize # getitem一次返回一个图像idx列表，列表长度为image_sampler_batchsize

        self.video_data_source_len = self.num_samples * self.num_replicas # video pad后的indices

    
        self.image_data_source_len =  image_sampler_batchsize * (1/video_data_step_ratio-1)  / video_sampler_batchsize * self.video_data_source_len
        print(f'{self.video_data_source_len=}, {self.image_data_source_len=}')

    def __iter__(self):

        # 创建索引
        video_indices = list(range(int(self.video_data_source_len)))
        image_indices = list(range(int(self.image_data_source_len)))
        # 如果需要 shuffle，则每个 epoch 打乱顺序
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)  # 使用epoch 作为种子
            indices = torch.randperm(self.pad_data_source_len, generator=g).tolist()

        # 每个gpu可以分到的indices个数, 一个idx本来是一个数字，现在变成了一个列表
        video_indices = video_indices[self.rank:self.total_size:self.num_replicas]
        len_of_video_indices = int(len(video_indices) // self.video_sampler_batchsize)
        
        new_indices= []
        for idx in range(len_of_video_indices):
            # video_idx   第一个是视频idx_list, 长度为video_sampler_batchsize
            s_videoidx1 =  idx * self.video_sampler_batchsize
            e_videoidx1 = ( idx+1) * self.video_sampler_batchsize
            item = video_indices[s_videoidx1:e_videoidx1] 
            item.append('video')
            new_indices.append(item)

            # image_idx  if self.video_data_step_ratio==4 后面3个是图像idx_list， 每一个长度为image_sampler_batchsize
            for image_idx in range(int(1/self.video_data_step_ratio)-1):

                s_imageidx =   ( idx * (int(1/self.video_data_step_ratio)-1) +  image_idx) * self.image_sampler_batchsize 
                e_imageidx =   s_imageidx +  self.image_sampler_batchsize

                item = image_indices[s_imageidx:e_imageidx] 
                item.append('image')
                new_indices.append(item)

        return iter(new_indices)

    def __len__(self):
        return  self.num_samples

    def set_epoch(self, epoch):
        # 在分布式训练时更新 epoch，从而保证每个 epoch 的 shuffle 不同
        self.epoch = epoch
        
class T2IV_dataset(Dataset):
    def __init__(self, args,  tokenizer, 
                 processor, temporal_downsample_factor, 
                 data_repeat=10, tokenizer_max_len=120,
                 latent_size=32, video_sampler_batchsize = None):
 
        self.video_data_root = args.video_data_root
        self.image_data_root = args.image_data_root
        self.num_frames = args.num_frames
        
 
        assert args.video_meta_info_file is not None
        self.video_meta_info = self.read_jsonfile(args.video_meta_info_file) # [:10]
        self.image_meta_info = self.read_jsonfile(args.image_meta_info_file)

        print(f'{args.video_meta_info_file=} is loaded successfully!')
        print(f'video_meta_info:{len(self.video_meta_info)}, image_meta_info:{len(self.image_meta_info)}!!!')
        # ========== new 
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_frames = args.num_frames
        self.temporal_downsample_factor = temporal_downsample_factor
        self.tokenizer_max_len =  tokenizer_max_len
        self.code_len = (latent_size ** 2) * (args.num_frames//4) # video vae tokens
        self.video_sampler_batchsize = video_sampler_batchsize 

        self.do_image_center_crop = args.do_image_center_crop # True
        self.image_crop_size = args.image_crop_size # 256


    def read_jsonfile(self, jsonfile):
        with open(jsonfile, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def __len__(self):
        # return self.training_sample_nums
        return len(self.video_meta_info)

    def read_video_frames(self, video_path, n_frames=16):
        """
        使用 decord 从视频中读取所有帧，并返回包含每一帧 PIL 图像的列表。
        这些帧的类型与 PIL.Image.open() 读取的帧类型相同.
        :param video_path: 视频文件路径
        :return: PIL 图像列表，每个元素是一个视频帧
        """
        # 使用 decord 打开视频文件
        vr = VideoReader(video_path, ctx=cpu(0))  # 在 CPU 上读取视频
        frames = []
        n_frames = min(n_frames, len(vr))
        # 遍历视频中的每一帧并保存
        # 遍历前 n_frames 帧
        for i in range(1, n_frames+1, 1):
            frame = vr[i]  # 通过索引获取帧
            img = Image.fromarray(frame.asnumpy())  # 转换为 PIL.Image.Image
            img_resized = img
            # img_resized = img.resize((256, 256))  # 调整大小为 256x256
            frames.append(img_resized)

        return frames

    def __getitem__(self, idx_list):
    
        
        assert type(idx_list) is list
        try:
            # video
            if idx_list[-1] == 'video':
                videos, texts = [], []
                video_idxs = idx_list[:-1]
                for video_idx in video_idxs:
                    video_idx = video_idx % len(self.video_meta_info)

                    text =  self.video_meta_info[video_idx]['cap'][-1]
                    images = self.get_video_data(video_idx) # (n_frame//4, 4, c, h, w)
                    texts.append(text)
                    videos.append(images)

                return dict(text=texts, visual_data=videos, data_type='video')

            # image
            elif idx_list[-1] == 'image':
                # import ipdb; ipdb.set_trace()
                images, texts = [], []
                image_idxs = idx_list[:-1]
                for image_idx in image_idxs:
                    # print(f'{image_idx=}')
                    image_idx = image_idx % len(self.image_meta_info)
                    # if image_idx > len(self.image_meta_info):
                    text =  self.image_meta_info[image_idx]['cap'][-1]
                    image = self.get_image_data(image_idx) # (n_frame=1, 1, c, h, w)
                    texts.append(text)
                    images.append(image)
                return dict(text=texts, visual_data=images, data_type='image')
            gc.collect()
       
        except Exception as e:
            print(e, '!!!!!!!!')
            random_idx = random.randint(0, self.__len__() - 1)
            idx = [ random_idx for i in range(self.video_sampler_batchsize)]
            idx.append('video')
            return self.__getitem__(idx)


    def resize_and_center_crop(self, image_path, output_size=512):

        # 打开图片
        image = Image.open(image_path)
        
        # 获取图片的原始尺寸
        width, height = image.size
        
        # 计算新的尺寸，保持宽高比
        if width < height:
            new_width = output_size
            new_height = int(output_size * height / width)
        else:
            new_width = int(output_size * width / height)
            new_height = output_size
        
        # 调整图片大小
        image = image.resize((new_width, new_height))
        
        # 计算中心裁剪的位置
        left = (new_width - output_size) // 2
        top = (new_height - output_size) // 2
        right = left + output_size
        bottom = top + output_size
        
        # 进行中心裁剪
        image = image.crop((left, top, right, bottom))
        
        return image


    def get_image_data(self, idx):
        image_path = os.path.join(self.image_data_root, self.image_meta_info[idx]['path'])
        if self.do_image_center_crop:
            image = self.resize_and_center_crop(image_path, output_size=self.image_crop_size)
        else:
            image =  Image.open(image_path) 
        image = self.processor(image, return_tensors="pt")["pixel_values"] 
        image = image.unsqueeze(0)  # (1, n_frame, c, h, w)
        return image

    def get_video_data(self, idx):
 
        video_path = os.path.join(self.video_data_root, self.video_meta_info[idx]['path'])
        video = self.read_video_frames(video_path, n_frames=self.num_frames)

        # import ipdb; ipdb.set_trace()
        images = self.processor(video, return_tensors="pt")["pixel_values"]
        images = images.unsqueeze(0) # (1, n_frame, c, h, w)
        # import ipdb; ipdb.set_trace()
        images = images.view(
                    -1,
                    self.temporal_downsample_factor,
                    *images.shape[2:],
                )  # === (n_frame//4, 4, c, h, w) ===
        return images

    """def get_video(self, idx):
        
        video_path = os.path.join(self.data_root, self.video_meta_info[idx]['path'])
        # filter video seconds less than 2s, start_idx=25, end_idx=25+self.num_frames
        video = self.decord_read(video_path)

        # import ipdb; ipdb.set_trace()
        video = self.transform(video)  # T C H W -> T C H W
        video = video.transpose(0, 1)  # T C H W -> C T H W


        text = random.choice(self.video_meta_info[idx]['cap'])
        # return dict(video=video, text=text)
        text = text_preprocessing(text)

        t5_file = self.get_npy_path(self.video_meta_info[idx])
        assert os.path.isfile(t5_file), 't5_file {} does not exist!'.format(t5_file)
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        t5_feat = torch.from_numpy(np.load(t5_file))
        
        t5_feat_len = t5_feat.shape[1] 
        feat_len = min(self.t5_feature_max_len, t5_feat_len)
        
 

        # import ipdb; ipdb.set_trace()
        t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
        emb_mask = torch.zeros((self.t5_feature_max_len,))
        emb_mask[-feat_len:] = 1
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length))
        T = self.t5_feature_max_len # 120
        attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
        eye_matrix = torch.eye(self.max_seq_length, self.max_seq_length)
        attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
        attn_mask = attn_mask.unsqueeze(0).to(torch.bool)

        # os.makedirs('visual_attnmask', exist_ok=True)
        # visualize_attn_mask(attn_mask, f'visual_attnmask/{idx}_attn_mask.png')

        valid = 1
        return dict(video=video, t5_feat_padding=t5_feat_padding, attn_mask=attn_mask, valid = torch.tensor(valid), text=text)

 """
    def decord_read(self, path):
        decord_vr = self.v_decoder(path)
        # Sampling video frames
        frame_indice = np.linspace(self.start_frame_ind, self.end_frame_ind - 1, self.num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



from transformers import AutoTokenizer
def tmp():
    import json
    # 读取json文件
    with open('/storage/zhubin/liuyihang/add_aes/output/sucai_aes.json', 'r') as file:
        data = json.load(file)
    # 提取前10个元素
    new_data = data[:1000]
    # 将新的列表写入新的json文件
    with open('/storage/zhubin/liuyihang/add_aes/output/sucai_aes_1000.json', 'w') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)

def build_t2iv(args, tokenizer, processor, temporal_downsample_factor, data_repeat, tokenizer_max_len):
    return T2IV_dataset(args, tokenizer, processor, temporal_downsample_factor, data_repeat, tokenizer_max_len)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # dataset & dataloader
    # parser.add_argument("--dataset", type=str, required=True)
    # parser.add_argument("--data", type=str, required='')
    parser.add_argument("--video_meta_info_file", type=str, default='/storage/zhubin/liuyihang/add_aes/output/sucai_aes_1000.json')
    # parser.add_argument("--sample_rate", type=int, default=1)
    # parser.add_argument("--cache_dir", type=str, required='')
    # parser.add_argument("--t5-model-path", type=str, default='./pretrained_models/t5-ckpt')
    # parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    # parser.add_argument("--max_height", type=int, default=1)
    # parser.add_argument("--max_width", type=int, default=1)
    parser.add_argument("--precision", type=str, default='bf16')
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--start_frame_ind", type=int, default=25)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--video_data_root", type=str, default='/storage/dataset')
    parser.add_argument("--image_data_root", type=str, default='/storage/zhubin/UniLLM')
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--t-downsample-size", type=int, choices=[4, 8], default=4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--t5-path", type=str, required=True)
    parser.add_argument("--vq_model", type=str, default="Emu3_VQ")
    parser.add_argument("--vq_repo", type=str, default="BAAI/Emu3-VisionTokenizer") 

    parser.add_argument("--image_meta_info_file", type=str, default='/storage/zhubin/UniLLM/tmp/image_data.json')
    parser.add_argument("--do_image_center_crop", action="store_true")
    parser.add_argument("--image_crop_size", type=int, default=256)
    
    args = parser.parse_args()

    init_distributed_mode(args)

    

    device='cuda'
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]


    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    from transformers import AutoTokenizer
    from autoregressive.models.tokenizer.emu3   import Emu3VisionVQImageProcessor
    

    from autoregressive.models.tokenizer.vq_models import VQ_models
    # import ipdb; ipdb.set_trace()
    vq_model  = VQ_models[args.vq_model]
    vq_model =  vq_model.from_pretrained(args.vq_repo, trust_remote_code=True, cache_dir='/storage/zhubin/UniLLM/cache_dir').eval().to(device)
    
    
    MODEL_HUB =   "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HUB, cache_dir="/storage/zhubin/UniLLM/cache_dir")
    
    # import ipdb; ipdb.set_trace()
    processor = Emu3VisionVQImageProcessor.from_pretrained("BAAI/Emu3-VisionTokenizer", trust_remote_code=True, cache_dir="/storage/zhubin/UniLLM/cache_dir")
    processor.max_pixels = 512*512
    processor.min_pixels = 256*256
    processor.size = {
        "max_pixels": 512*512,
        "min_pixels": 256*256
    }
    temporal_downsample_factor = 4
    
    video_sampler_batchsize = 2
    dataset = T2IV_dataset(args=args, \
                          tokenizer=tokenizer, 
                          processor=processor, 
                          temporal_downsample_factor=temporal_downsample_factor, 
                          data_repeat=10, 
                          tokenizer_max_len=120,
                          video_sampler_batchsize = video_sampler_batchsize
                          ) # 
    
    rank = dist.get_rank()
    # args.world_size = dist.get_world_size()
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=rank,
    #     shuffle=True,
    #     seed=args.global_seed
    # )

    sampler = SimpleDistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=rank,
        video_sampler_batchsize=2, image_sampler_batchsize=3, video_data_step_ratio=1/4,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # import ipdb; ipdb.set_trace()
    for idx, sample in enumerate(dataloader):

        # import ipdb; ipdb.set_trace()
        # print(rank, sample)
        # continue
        texts = sample['text'] # list      sample['data_type'][0] 
        visual_data = torch.cat(sample['visual_data'], dim=0) # (bs, n_frame//4, 4, c, h, w)
        data_type = sample['data_type'][0]
        print(rank,idx, visual_data.shape, data_type)
        # continue
        with torch.no_grad():
            # encode      visual_data_flat = visual_data.reshape(b * n, c, h, w)
            (b, n, t, c, h ,w) = visual_data.shape # [2, 8, 4, 3, 512, 512] or [2, 8, 4, 3, 512, 512]
            if data_type == 'video':
                visual_data_flat = visual_data.reshape(b * n, t, c, h, w) # [16, 4, 3, 512, 512]  or  [16, 4, 3, 256, 256] 
            elif data_type == 'image':
                visual_data = visual_data[:,0]
                visual_data_flat = visual_data.reshape(b * n, c, h, w)
            codes = vq_model.encode( visual_data_flat.to(device)) # [16, 1, 64, 64]  or  [16, 1, 32, 32]
            codes = codes.reshape(b, -1) # torch.Size([2, 32768])  or  [2, 8192]
            print(codes.shape)

            # bn, t_, h_ , w_ = codes.shape
            # codes = codes.reshape(b, n, t_, h_, w_) # [2, 8, 1, 64, 64]
 

"""
VIDEO_DATA_FILE='/storage/zhubin/UniLLM/dataset/sucai_final_720p_2490942.json'
IMAGE_DATA_FILE='/storage/zhubin/UniLLM/dataset/recap_final_512+_3796513.json'


export master_addr=127.0.0.1
export master_port=29505
export CUDA_VISIBLE_DEVICES=7

cd  /storage/zhubin/UniLLM
source  /storage/miniconda3/etc/profile.d/conda.sh 
conda activate  


export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

HF_DATASETS_OFFLINE=1   torchrun  --nnodes=1  --nproc_per_node=1  \
--master_addr=$master_addr --master_port=$master_port \
dataset/t2iv.py \
--video_meta_info_file $VIDEO_DATA_FILE \
--num_frames 16 \
--video_data_root  /storage/dataset \
--image_data_root  /storage/dataset/recap_datacomp_1b_data/output  \
--t5-path  /storage/zhubin/LlamaGen/dataset/storage_datasets_npy \
--num-workers 0 \
--vq_model   Emu3_VQ \
--vq_repo  BAAI/Emu3-VisionTokenizer \
--image_meta_info_file  $IMAGE_DATA_FILE \
--image_crop_size   256 \
--do_image_center_crop  

 
"""