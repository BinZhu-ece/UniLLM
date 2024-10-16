DATA_FILE='/storage/zhubin/liuyihang/add_aes/output/sucai_aes_1000.json'

export CUDA_VISIBLE_DEVICES=7
export master_addr=127.0.0.1
export master_port=29502
export CUDA_VISIBLE_DEVICES=7


source  /storage/miniconda3/etc/profile.d/conda.sh 
conda activate motionctrl

cd /storage/zhubin/LlamaGen 
conda activate motionctrl
torchrun --nnodes=1  --nproc_per_node=1 --master_addr=$master_addr --master_port=$master_port \
dataset/t2i.py \
--data-path /storage/zhubin/LlamaGen/dataset/Image_Datasets/  \
--t5-feat-path /storage/zhubin/LlamaGen/dataset/Image_Datasets/  \
--short-t5-feat-path /storage/zhubin/LlamaGen/dataset/Image_Datasets/  \
--dataset t2i \
--image-size 256 
 


"""
bash /storage/zhubin/LlamaGen/scripts/dataset/t2v_dataset.bash
"""