# VideoMAE Installation

The codebase is mainly built with following libraries:

- Python 3.6 or higher
- [PyTorch](https://pytorch.org/) and [torchvision](https://github.com/pytorch/vision). <br>
  We can successfully reproduce the main results in two settings:<br>
  Tesla **A100** (40G): CUDA 11.1 + PyTorch 1.8.0 + torchvision 0.9.0
  Tesla **V100** (32G): CUDA 10.1 + PyTorch 1.6.0 + torchvision 0.7.0
- [timm==0.4.8/0.4.12](https://github.com/rwightman/pytorch-image-models)
- [deepspeed==0.5.8](https://github.com/microsoft/DeepSpeed)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [decord](https://github.com/dmlc/decord)
- [einops](https://github.com/arogozhnikov/einops)

