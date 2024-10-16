# !/bin/bash



# debug and save_train_video_latent

cd  /storage/zhubin/UniLLM
DATA_FILE='/storage/zhubin/video_statistics_data/task1.5/Final_format_dataset_data_v2/step1.5_storyblocks_final_1270947_filter_1031888.json'

nnodes=1
nproc_per_node=8
export master_addr=127.0.0.1
export master_port=29502
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


source  /storage/miniconda3/etc/profile.d/conda.sh 
conda activate 

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node  \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_t2v.py \
--data-path None \
--video_meta_info_file $DATA_FILE \
--data_root  /storage/dataset  \
--cloud-save-path ./cloud_path_t2v  \
--global-batch-size $(( 4 * $nproc_per_node )) \
--epochs  5 \
--dataset t2v \
--num-workers 16  \
--log-every 30  \
--ckpt-every  1000  \
--results-dir results/1.5B-4f-256px \
--num-frames 4 \
--llm_model_hub Qwen/Qwen2.5-1.5B \
--tokenizer_max_len 512 \
--vq-model   Emu3_VQ \
--vq-repo  BAAI/Emu3-VisionTokenizer \
--gradient_accumulation_steps 8 \
--no-compile


 
