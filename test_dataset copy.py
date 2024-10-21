import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import math

class SimpleDistributedSampler(Sampler):
    def __init__(self, data_source, num_replicas=None, rank=None, shuffle=False, 
                epoch=1, video_sampler_batchsize=2, image_sampler_batchsize=3, video_data_step_ratio=1/4,
                total_video_nums=0, total_video_nums=0):
        
        

        self.data_source = total_video_nums  
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

        self.total_video_nums = total_video_nums
        # 如果image的idx超过了图像总量，可以%取余
        self.total_image_nums = total_video_nums * ( int(1/video_data_step_ratio-1)*image_sampler_batchsize / (1*video_sampler_batchsize)  )  


    def __iter__(self):


        # 创建索引
        indices = list(range(len(self.total_video_nums)))

        # 如果需要 shuffle，则每个 epoch 打乱顺序
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)  # 使用epoch 作为种子
            indices = torch.randperm(len(self.data_source), generator=g).tolist()

        # 扩展到可以整除 num_replicas 的长度
        indices += indices[:(self.total_size - len(indices))]



        # 视频batch和图像batch完成一次交替需要的idx个数，与sampler_batchsize, video_data_step_ratio, image_scale_bs都有关
        idx_nums_of_data_type_change = self.video_sampler_batchsize * 1 + self.image_sampler_batchsize * (1/self.video_data_step_ratio - 1)
        idx_nums_of_data_type_change = int(idx_nums_of_data_type_change)


        # 每个gpu可以分到的indices个数, 一个idx本来是一个数字，现在变成了一个列表
        indices = indices[self.rank:self.total_size:self.num_replicas]
        len_of_indices = int(len(indices) // idx_nums_of_data_type_change)
        new_indices= []

        for i in range(len_of_indices):

            # video_idx   第一个是视频idx_list, 长度为video_sampler_batchsize
            s_videoidx1 = i * idx_nums_of_data_type_change
            e_videoidx1 = s_videoidx1 + self.video_sampler_batchsize

            item = indices[s_videoidx1:e_videoidx1] 
            item.append('video')
            new_indices.append(item)

            # image_idx  if self.video_data_step_ratio==4 后面3个是图像idx_list， 每一个长度为image_sampler_batchsize
            for i in range(int(1/self.video_data_step_ratio)-1):
                s_imageidx = e_videoidx1 + self.image_sampler_batchsize * i
                e_imageidx = s_imageidx + self.image_sampler_batchsize

                item = indices[s_imageidx:e_imageidx] 
                item.append('image')
                new_indices.append(item)

        return iter(new_indices)

    def __len__(self):
        return 2 #self.num_samples

    def set_epoch(self, epoch):
        # 在分布式训练时更新 epoch，从而保证每个 epoch 的 shuffle 不同
        self.epoch = epoch
# 定义一个简单的 Dataset，直接返回索引值


import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group(backend="nccl")

# 定义一个简单的数据集
class SimpleDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return idx

# 获取进程 rank 和进程总数
rank = dist.get_rank()
world_size = dist.get_world_size()

# 创建数据集实例
dataset = SimpleDataset()

# 创建自定义分布式 Sampler
sampler = SimpleDistributedSampler(dataset, num_replicas=world_size, rank=rank)

# 使用 DataLoader 加载数据，传入自定义的分布式 Sampler
dataloader = DataLoader(dataset, sampler=sampler, batch_size=1, drop_last=True, num_workers=4)

# 迭代并打印每个进程处理的 batch 数据
for batch in dataloader:
    print(f"Rank {rank}, Batch: {batch}")

# 记得在最后销毁进程组
dist.destroy_process_group()


"""
cd  /storage/zhubin/UniLLM
nnodes=1
nproc_per_node=2
export master_addr=127.0.0.1
export master_port=29502
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source  /storage/miniconda3/etc/profile.d/conda.sh 
conda activate 

HF_DATASETS_OFFLINE=1 torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node  \
--master_addr=$master_addr --master_port=$master_port \
test_dataset.py  

"""