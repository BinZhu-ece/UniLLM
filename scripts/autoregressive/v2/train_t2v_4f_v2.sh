# !/bin/bash



# debug and save_train_video_latent

cd  /storage/zhubin/UniLLM
VIDEO_DATA_FILE='/storage/zhubin/video_statistics_data/task1.5/Final_format_dataset_data_v2/step1.5_storyblocks_final_1270947_filter_1031888.json'
IMAGE_DATA_FILE='/storage/anno_jsons/tuzhan_mj_1712571_resolution.json'
# ,
nnodes=1
nproc_per_node=1
batchsize_per_gpu=4


export master_addr=127.0.0.1
export master_port=29504
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7


source  /storage/miniconda3/etc/profile.d/conda.sh 
conda activate 

HF_DATASETS_OFFLINE=1 torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node  \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_t2iv.py \
--data-path None \
--video_meta_info_file $VIDEO_DATA_FILE \
--image_meta_info_file  $IMAGE_DATA_FILE \
--video_data_root  /storage/dataset \
--image_data_root  /storage/dataset/image/tuzhan_mj  \
--cloud-save-path ./cloud_path_t2v  \
--global-batch-size $(( $batchsize_per_gpu * $nproc_per_node )) \
--epochs  5 \
--dataset t2iv \
--num-workers 16  \
--log-every 30  \
--ckpt-every  10000  \
--results-dir results/1.5B-4f-256px-leftpad-attnmatrix \
--num-frames 16 \
--llm_model_hub Qwen/Qwen2.5-1.5B \
--tokenizer_max_len 512 \
--vq-model   Emu3_VQ \
--vq-repo  BAAI/Emu3-VisionTokenizer \
--gradient_accumulation_steps 8 \
--video_sampler_batchsize 2 \
--image_sampler_batchsize 5 \
--no-compile


 
