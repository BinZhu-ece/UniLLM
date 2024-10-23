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


class T2V_dataset(Dataset):
    def __init__(self, args,  tokenizer, 
                 processor, temporal_downsample_factor, 
                 data_repeat=10, tokenizer_max_len=120,
                 latent_size=32):

        self.data_root = args.data_root
        self.num_frames = args.num_frames
        # self.transform = transform
        # self.t5_xxl = t5_xxl
        # self.t5_path = args.t5_path
        # self.model_max_length = args.model_max_length
        # self.v_decoder = DecordInit()
        # video_meta_info_file
        self.video_meta_info = self.read_jsonfile(args.video_meta_info_file)*data_repeat
        print(f'Data repeat {data_repeat} times during initialize dataset!')
        print(f'{args.video_meta_info_file=} is loaded successfully!')
        # self.start_frame_ind = args.start_frame_ind # start from 1 s
        # self.end_frame_ind = args.start_frame_ind + args.num_frames # 

        # =================== !!!!!!!!!!!!!
        # downsample_size:32,  video_t = len(frames)//4+1
        # latent_size = args.image_size // args.downsample_size 
        # self.code_len = (latent_size ** 2) * (args.num_frames//4) # video vae tokens
        # self.t5_feature_max_len = 120
        # self.t5_feature_dim = 2048
        # self.max_seq_length = self.t5_feature_max_len + self.code_len


        # ========== new 
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_frames = args.num_frames
        self.temporal_downsample_factor = temporal_downsample_factor
        self.tokenizer_max_len =  tokenizer_max_len
        self.code_len = (latent_size ** 2) * (args.num_frames//4) # video vae tokens


    def read_jsonfile(self, jsonfile):
        with open(jsonfile, 'r', encoding='utf-8') as f:
            return json.load(f)

    def __len__(self):
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
            img_resized = img.resize((256, 256))  # 调整大小为 256x256
            frames.append(img_resized)

        return frames

    def __getitem__(self, idx):

        try:
            
            # caption 
            text = random.choice(self.video_meta_info[idx]['cap'])
            inputs = self.tokenizer(
                text, return_tensors="pt",
                padding="max_length",  # 使用最大长度进行填充
                max_length=self.tokenizer_max_len,    # 这里替换为你想要的固定长度
                truncation=True        # 如果文本超过最大长度则截断
                )
            input_ids = inputs['input_ids'] # === (1, max_length) ===
            attention_mask = inputs['attention_mask'] # === (1, max_length) ===
            # attention_mask for (text & video)
            T  = self.tokenizer_max_len
            attention_mask = torch.cat(
                        (attention_mask, torch.ones(1, self.code_len)), 
                        dim=1
                )

            """
            T  = self.tokenizer_max_len
            attn_mask = torch.tril(torch.ones(( T + self.code_len, 
                                                T + self.code_len), 
                                               dtype=torch.long))
            attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
            eye_matrix = torch.eye( T + self.code_len,  T + self.code_len)
            attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
            attn_mask = attn_mask.unsqueeze(0).to(torch.bool) # === (1, T+code_len, T+code_len) ==="""

            # video
            video_path = os.path.join(self.data_root, self.video_meta_info[idx]['path'])
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
            gc.collect()
            return dict(input_ids=input_ids, attention_mask=attention_mask, video_data=images)
        except Exception as e:
            print(e, '!!!!!!!!')
            return self.__getitem__(random.randint(0, self.__len__() - 1))


    def get_npy_path(self, item):
        video_rela_path = item['path']
        dir_name = os.path.dirname(video_rela_path)
        filename = os.path.splitext(os.path.basename(video_rela_path))[0]    
        npy_path = os.path.join(self.t5_path, dir_name, '{}.npy'.format(filename))
        return npy_path


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

# from dataset.transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo
# from CausalVideoVAE.causalvideovae.model import ae_norm, ae_denorm
from torchvision.transforms import Lambda
ae_norm = {
    'CausalVAEModel_D4_2x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVAEModel_D8_2x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVAEModel_D4_4x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVAEModel_D8_4x8x8': Lambda(lambda x: 2. * x - 1.),
}
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

def build_t2v(args, tokenizer, processor, temporal_downsample_factor, data_repeat, tokenizer_max_len):
    return T2V_dataset(args, tokenizer, processor, temporal_downsample_factor, data_repeat, tokenizer_max_len)


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
    parser.add_argument("--data_root", type=str, default='/storage/dataset')

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--t-downsample-size", type=int, choices=[4, 8], default=4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--t5-path", type=str, required=True)
    parser.add_argument("--vq_model", type=str, default="Emu3_VQ")
    parser.add_argument("--vq_repo", type=str, default="BAAI/Emu3-VisionTokenizer") 
    args = parser.parse_args()

    init_distributed_mode(args)

    # resize = [CenterCropResizeVideo((args.max_height, args.max_width)), ]
    # norm_fun = ae_norm[args.ae]
    # norm_fun = Lambda(lambda x: 2. * x - 1.)
    # transform = transforms.Compose([
    #     ToTensorVideo(),
    #     *resize, 
    #     # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
    #     norm_fun
    # ])
    from dataset.augmentation import center_crop_arr
    # transform = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    # ])

    # tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir)
    # assert os.path.exists(args.t5_model_path)
    device='cuda'
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]


    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    from transformers import AutoTokenizer
    from autoregressive.models.tokenizer.emu3   import Emu3VisionVQImageProcessor



    from autoregressive.models.tokenizer.vq_models import VQ_models
    vq_model, processor = VQ_models[args.vq_model]
    vq_model =  vq_model.from_pretrained(args.vq_repo, trust_remote_code=True, cache_dir='./cache_dir').eval().to(device)


    MODEL_HUB =   "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HUB, cache_dir="/storage/zhubin/UniLLM/cache_dir")

    # import ipdb; ipdb.set_trace()
    processor = Emu3VisionVQImageProcessor.from_pretrained("BAAI/Emu3-VisionTokenizer", trust_remote_code=True, cache_dir="/storage/zhubin/UniLLM/cache_dir")
    # processor.max_pixels = 512*512
    # processor.min_pixels = 256*256
    # processor.size = {
    #     "max_pixels": 512*512,
    #     "min_pixels": 256*256
    # }
    temporal_downsample_factor = 4


    dataset = T2V_dataset(args=args, \
                          tokenizer=tokenizer, 
                          processor=processor, 
                          temporal_downsample_factor=temporal_downsample_factor, 
                          data_repeat=10, 
                          tokenizer_max_len=120) # 

    rank = dist.get_rank()
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )


    # import ipdb; ipdb.set_trace()
    for idx, sample in enumerate(dataloader):

        # torch.Size([2, 1, 120]) torch.Size([2, 1, 120]) torch.Size([2, 8, 4, 3, 512, 512])
        print(sample['input_ids'].shape, sample['attention_mask'].shape, sample['video_data'].shape)

        with torch.no_grad():
            # encode
            (b, n, t, c, h ,w) = sample['video_data'].shape # [2, 8, 4, 3, 512, 512] or [2, 8, 4, 3, 512, 512]
            video_data_flat = sample['video_data'].reshape(b * n, t, c, h, w) # [16, 4, 3, 512, 512]  or  [16, 4, 3, 256, 256] 
            codes = vq_model.encode(video_data_flat.to(device)) # [16, 1, 64, 64]  or  [16, 1, 32, 32]
            codes = codes.reshape(b, -1) # torch.Size([2, 32768])  or  [2, 8192]
            print(codes.shape)

            # bn, t_, h_ , w_ = codes.shape
            # codes = codes.reshape(b, n, t_, h_, w_) # [2, 8, 1, 64, 64]


"""
DATA_FILE='/storage/zhubin/video_statistics_data/task1.5/Final_format_dataset_data_v2/step1.5_coverr_final_3002.json'
 
export master_addr=127.0.0.1
export master_port=29507
export CUDA_VISIBLE_DEVICES=6
cd  /storage/zhubin/UniLLM
source  /storage/miniconda3/etc/profile.d/conda.sh 
conda activate  
 
torchrun \
--nnodes=1  --nproc_per_node=1  \
--master_addr=$master_addr --master_port=$master_port \
dataset/t2v.py \
--video_meta_info_file $DATA_FILE \
--num_frames 8 \
--data_root  /storage/dataset \
--t5-path  /storage/zhubin/LlamaGen/dataset/storage_datasets_npy \
--num-workers 0 \
--vq_model   Emu3_VQ \
--vq_repo  BAAI/Emu3-VisionTokenizer 
"""