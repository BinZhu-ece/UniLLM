import torch
from torch.utils.data import DataLoader, Dataset
import os

# 假设我们有图片和视频的路径列表
image_paths = [f"image_{i}.jpg" for i in range(9000)]  # 模拟图片路径
video_paths = [f"video_{i}.mp4" for i in range(1000)]  # 模拟视频路径

# 创建自定义数据集用于加载路径
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.image_paths[idx]

class VideoDataset(Dataset):
    def __init__(self, video_paths):
        self.video_paths = video_paths

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        return self.video_paths[idx]

# 模型检查点保存函数
def save_checkpoint(step, image_iter_state, video_iter_state, ckpt_path="./tmp-checkpoint.pth"):
    torch.save({
        'step': step,
        'image_iter_state': image_iter_state,
        'video_iter_state': video_iter_state
    }, ckpt_path)
    print(f"Checkpoint saved at step {step}.")

# 模型检查点加载函数
def load_checkpoint(ckpt_path="checkpoint.pth"):
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        step = checkpoint['step']
        image_iter_state = checkpoint['image_iter_state']
        video_iter_state = checkpoint['video_iter_state']
        print(f"Checkpoint loaded from step {step}.")
        return step, image_iter_state, video_iter_state
    else:
        return 0, None, None  # 如果没有checkpoint，默认从0开始

# 加载训练数据
def get_dataloader():
    image_loader = DataLoader(ImageDataset(image_paths), batch_size=4, shuffle=False)
    video_loader = DataLoader(VideoDataset(video_paths), batch_size=4, shuffle=False)
    return image_loader, video_loader

# 主训练函数
def train(total_steps=10000, resume=False, ckpt_path="checkpoint.pth"):
    image_loader, video_loader = get_dataloader()

    # 初始化迭代器
    image_iter = iter(image_loader)
    video_iter = iter(video_loader)

    import ipdb; ipdb.set_trace()
    # 加载检查点状态
    if resume:
        step, image_iter_state, video_iter_state = load_checkpoint(ckpt_path)
        if image_iter_state:
            image_iter = torch.utils.data.DataLoader.load_state_dict(image_iter_state)
        if video_iter_state:
            video_iter = torch.utils.data.DataLoader.load_state_dict(video_iter_state)
    else:
        step = 0

    # 迭代器在 PyTorch 中不直接支持保存状态，因此在resume时需要重构迭代器
    for step in range(step, total_steps):
        if step % 10 < 9:  # 前 9 个 step 处理图像
            try:
                image_batch = next(image_iter)
            except StopIteration:
                image_iter = iter(image_loader)
                image_batch = next(image_iter)
            print(f"Step {step}: Processing image batch - {image_batch}")
        else:  # 第 10 个 step 处理视频
            try:
                video_batch = next(video_iter)
            except StopIteration:
                video_iter = iter(video_loader)
                video_batch = next(video_iter)
            print(f"Step {step}: Processing video batch - {video_batch}")

        # 每 100 步保存一次检查点
        if step % 10 == 0:
            save_checkpoint(step, image_iter.state_dict(), video_iter.state_dict(), ckpt_path)

if __name__ == "__main__":
    # 调用训练函数，使用resume=True来恢复
    train(total_steps=10000, resume=False, ckpt_path="checkpoint.pth")
