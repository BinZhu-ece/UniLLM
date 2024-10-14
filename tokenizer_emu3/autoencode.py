# -*- coding: utf-8 -*-
import torch
import os
import os.path as osp
import decord
from decord import VideoReader, cpu
from PIL import Image

from transformers import AutoModel, AutoImageProcessor

MODEL_HUB = "BAAI/Emu3-VisionTokenizer"




def read_video_frames(video_path, n_frames=16):
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
    for i in range(1, n_frames, 5):
        frame = vr[i]  # 通过索引获取帧
        img = Image.fromarray(frame.asnumpy())  # 转换为 PIL.Image.Image
        img_resized = img.resize((256, 256))  # 调整大小为 256x256
        frames.append(img_resized)

    return frames

videopath = '/storage/zhubin/UniLLM/video_samples/c2i7cCwRuw4_segment_1.mp4'
video = read_video_frames(videopath, n_frames=32)


model = AutoModel.from_pretrained(MODEL_HUB, trust_remote_code=True, cache_dir='./cache').eval().cuda()
processor = AutoImageProcessor.from_pretrained(MODEL_HUB, trust_remote_code=True)
images = processor(video, return_tensors="pt")["pixel_values"]
images = images.unsqueeze(0).cuda()

# image autoencode
image = images[:, 0]
print(image.shape)
with torch.no_grad():
    # encode
    codes = model.encode(image)
    # decode
    recon = model.decode(codes)

recon = recon.view(-1, *recon.shape[2:])
recon_image = processor.postprocess(recon)["pixel_values"][0]
recon_image.save("recon_image.png")


# video autoencode
images = images.view(
    -1,
    model.config.temporal_downsample_factor,
    *images.shape[2:],
)

print(images.shape)
with torch.no_grad():
    # encode
    codes = model.encode(images)
    # decode
    recon = model.decode(codes)

recon = recon.view(-1, *recon.shape[2:])
recon_images = processor.postprocess(recon)["pixel_values"]
for idx, im in enumerate(recon_images):
    im.save(f"recon_video_{idx}.png")


"""

cd  /storage/zhubin/UniLLM/ 
conda activate emu3
python tokenizer_emu3/autoencode.py

"""