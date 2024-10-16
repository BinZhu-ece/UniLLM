DATA_FILE='/storage/zhubin/LlamaGen/dataset/Image_Datasets/flowers/meta_data.json'

# export CUDA_VISIBLE_DEVICES=7
export master_addr=127.0.0.1
export master_port=29508
# export CUDA_VISIBLE_DEVICES=7

cd  /storage/zhubin/LlamaGen

source  /storage/miniconda3/etc/profile.d/conda.sh 
conda activate motionctrl


# conda activate motionctrl
torchrun \
--nnodes=1  --nproc_per_node=1 \
--master_addr=$master_addr --master_port=$master_port \
dataset/t2v.py \
--t5-model-path  pretrained_models/t5-ckpt \
--t5-model-type  flan-t5-xl \
--video_meta_info_file $DATA_FILE \
--max_height 480 \
--max_width 640 \
--model_max_length 512 \
--start_frame_ind 1 \
--num_frames 1 \
--t5-path  /storage/zhubin/LlamaGen/dataset/storage_datasets_npy/flowers \
--num-workers 16 \
--data_root  /storage/zhubin/LlamaGen/dataset/Image_Datasets/flowers 

"""
bash /storage/zhubin/LlamaGen/scripts/dataset/t2v_dataset.bash
"""