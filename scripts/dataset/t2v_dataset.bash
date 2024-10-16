DATA_FILE='/storage/zhubin/video_statistics_data/task1.5/Final_format_dataset_data_v2/step1.5_coverr_final_3002.json'

export CUDA_VISIBLE_DEVICES=7
export master_addr=127.0.0.1
export master_port=29505
export CUDA_VISIBLE_DEVICES=7

cd  /storage/zhubin/UniLLM

source  /storage/miniconda3/etc/profile.d/conda.sh 
conda activate  

 
torchrun \
--nnodes=1  --nproc_per_node=1  \
--master_addr=$master_addr --master_port=$master_port \
dataset/t2v.py \
--video_meta_info_file $DATA_FILE \
--model_max_length 512 \
--start_frame_ind 25 \
--num_frames 32 \
--data_root  /storage/dataset \
--t5-path  /storage/zhubin/LlamaGen/dataset/storage_datasets_npy \
--num-workers 0 \
--vq_model   Emu3_VQ \
--vq_repo  BAAI/Emu3-VisionTokenizer  \

 
"""
bash /storage/zhubin/UniLLM/scripts/dataset/t2v_dataset.bash
"""