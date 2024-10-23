# !/bin/bash



# debug and save_train_video_latent

cd  /storage/zhubin/UniLLM
VIDEO_DATA_FILE='/storage/zhubin/UniLLM/dataset/sucai_final_720p_2490942.json'
IMAGE_DATA_FILE='/storage/zhubin/UniLLM/dataset/recap_final_512+_3796513.json'
IMAGE_DATA_ROOT='/storage/dataset/recap_datacomp_1b_data/output'
VIDEO_DATA_ROOT='/storage/dataset'
NUM_FRAMES=16
VIDEO_SAMPLER_BATCHSIZE=1
IMAGE_SAMPLER_BATCHSIZE=6
IMAGE_CROP_SIZE=256
VIDEO_DATA_STEP_RATIO=0.1

nnodes=1
nproc_per_node=8
batchsize_per_gpu=4


export master_addr=127.0.0.1
export master_port=29504
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


source  /storage/miniconda3/etc/profile.d/conda.sh 
conda activate 
HF_DATASETS_OFFLINE=1 torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node  \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_t2iv.py \
--data-path None \
--video_meta_info_file $VIDEO_DATA_FILE \
--image_meta_info_file  $IMAGE_DATA_FILE \
--video_data_root  $VIDEO_DATA_ROOT  \
--image_data_root  $IMAGE_DATA_ROOT  \
--cloud-save-path ./cloud_path_t2v  \
--global-batch-size $(( $batchsize_per_gpu * $nproc_per_node )) \
--epochs  5 \
--dataset t2iv \
--num-workers 16  \
--log-every 30  \
--ckpt-every  10000  \
--results-dir results/1.5B-4f-256px-leftpad-attnmatrix \
--num-frames $NUM_FRAMES  \
--llm_model_hub Qwen/Qwen2.5-1.5B \
--tokenizer_max_len 512 \
--vq-model   Emu3_VQ \
--vq-repo  BAAI/Emu3-VisionTokenizer \
--gradient_accumulation_steps 8 \
--video_sampler_batchsize $VIDEO_SAMPLER_BATCHSIZE \
--image_sampler_batchsize $IMAGE_SAMPLER_BATCHSIZE \
--image_crop_size   $IMAGE_CROP_SIZE  \
--video_data_step_ratio   $VIDEO_DATA_STEP_RATIO \
--do_image_center_crop  \
--no-compile


 
