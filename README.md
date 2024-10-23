# UniLLM


<div align="center">

[![demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Online_Demo-blue)](https://huggingface.co/spaces/FoundationVision/LlamaGen)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2406.06525-b31b1b.svg)](https://arxiv.org/abs/2406.06525)&nbsp;
[![project page](https://img.shields.io/badge/Project_page-More_visualizations-green)](https://peizesun.github.io/llamagen/)&nbsp;

</div>


<p align="center">
<img src="assets/teaser.jpg" width=95%>
<p>



This repo contains pre-trained model weights and training/sampling PyTorch(torch>=2.1.0) codes used in

> [**Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation**](https://arxiv.org/abs/2406.06525)<br>
> [Peize Sun](https://peizesun.github.io/), [Yi Jiang](https://enjoyyi.github.io/), [Shoufa Chen](https://www.shoufachen.com/), [Shilong Zhang](https://jshilong.github.io/), [Bingyue Peng](), [Ping Luo](http://luoping.me/), [Zehuan Yuan](https://shallowyuan.github.io/)
> <br>HKU, ByteDance<br>

You can find more visualizations on [![project page](https://img.shields.io/badge/Project_page-More_visualizations-green)](https://peizesun.github.io/llamagen/)

## 🔥 Update
- [2024.10.23] Code are preparing !

```
Huggingface download: https://huggingface.co/Qwen/Qwen2.5-1.5B

```

 
## 🚀 Text-conditional image generation
### VQ-VAE models
Method | params | tokens | data | weight
--- |:---:|:---:|:---:|:---:
vq_ds16_t2i | 72M | 16x16 | LAION COCO (50M) + internal data (10M) | [vq_ds16_t2i.pt](https://huggingface.co/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt)

### AR models
Method | params | tokens | data | weight 
--- |:---:|:---:|:---:|:---:
LlamaGen-XL  | 775M | 16x16 | LAION COCO (50M) | [t2i_XL_stage1_256.pt](https://huggingface.co/peizesun/llamagen_t2i/resolve/main/t2i_XL_stage1_256.pt)
LlamaGen-XL  | 775M | 32x32 | internal data (10M) | [t2i_XL_stage2_512.pt](https://huggingface.co/peizesun/llamagen_t2i/resolve/main/t2i_XL_stage2_512.pt)

### Demo
Before running demo, please refer to [language readme](language/README.md) to install the required packages and language models.  

Please download models, put them in the folder `./pretrained_models`, and run
```
python3 autoregressive/sample/sample_t2i.py --vq-ckpt ./pretrained_models/vq_ds16_t2i.pt --gpt-ckpt ./pretrained_models/t2i_XL_stage1_256.pt --gpt-model GPT-XL --image-size 256
# or
python3 autoregressive/sample/sample_t2i.py --vq-ckpt ./pretrained_models/vq_ds16_t2i.pt --gpt-ckpt ./pretrained_models/t2i_XL_stage2_512.pt --gpt-model GPT-XL --image-size 512
```
The generated images will be saved to `sample_t2i.png`.

### Local Gradio Demo



## ⚡ Serving
We use serving framework [vLLM](https://github.com/vllm-project/vllm) to enable higher throughput. Please refer to [serving readme](autoregressive/serve/README.md) to install the required packages.  
```
python3 autoregressive/serve/sample_c2i.py --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_XXL_384.pt --gpt-model GPT-XXL --from-fsdp --image-size 384
```
The generated images will be saved to `sample_c2i_vllm.png`.


## Getting Started
See [Getting Started](GETTING_STARTED.md) for installation, training and evaluation.


## License
The majority of this project is licensed under MIT License. Portions of the project are available under separate license of referred projects, detailed in corresponding files.


## BibTeX
```bibtex
@article{sun2024autoregressive,
  title={Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation},
  author={Sun, Peize and Jiang, Yi and Chen, Shoufa and Zhang, Shilong and Peng, Bingyue and Luo, Ping and Yuan, Zehuan},
  journal={arXiv preprint arXiv:2406.06525},
  year={2024}
}
```
