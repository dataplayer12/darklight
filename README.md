# efficient-knowledge-distillation
Efficient KD using TensorRT inference on teacher network

# Background
Knowledge Distillation (KD) refers to the practice of using the outputs of a large teacher network train a (usually) smaller student network. This project leverages TensorRT to accelerate this process. It is common practice in KD, especially [dark knowledge](https://arxiv.org/abs/1503.02531) type techniques to pre-compute the logits from the teacher network and save them to disk. For training the student network, the pre-computed logits are used as is to teach the student. This saves GPU resources as one does not need to load the large teacher network to GPU memory during training.

# Problem
In [A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237), Beyer et. al. find that pre-computing logits is sub-optimal and hurts performance. The transformations applied to input (ex. blur, color jitter to images) are different between teacher and student, so the teacher logits do not correspond to the inputs seen by the student. Instead, for optimal knowdge distillation, the outputs from the teacher network should be computed exactly on the same input seen by the student. 

How to achieve the best techer inference and student training performance on a GPU?

# Solution
- Use TensorRT to set up an inference engine and perform blazing fast inference
- Use logits from TensorRT inference to train the student network.

# Environment

- Recommended method
This project uses pytorch CUDA, tensorrt>=8, opencv and pycuda. The recommended way to get all these is to use an NGC docker container with a recent version of PyTorch.

```Shell
sudo docker run -it --ipc=host --net=host --gpus all nvcr.io/nvidia/pytorch:22.08-py3 /bin/bash
#if you want to load an external disk to the container, use the --volume switch

#Once the container is up and running, install pycuda
pip install pycuda
git clone https://github.com/dataplayer12/efficient-knowledge-distillation.git
cd efficient-knowledge-distillation

#Test tensorRT engine with
python3 testtrt.py
```

- Custom env

If you want to use your own environment with PyTorch, you need to get TensorRT and pycuda.

Follow the [official guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip) to download TensorRT deb file and install it with the script provided in this repo. Finally install pycuda
```Shell
git clone https://github.com/dataplayer12/efficient-knowledge-distillation.git
cd efficient-knowledge-distillation
bash install_trt.sh
# if needed modify the version of deb file in the script before running.
# This script will also install pycuda
# this might fail for a number of reasons which is why NGC container is recommended

python3 testtrt.py #test if everything works
```

# Status
- [x] The core TensorRT functionality works well (can also be used for pure inference) 
- [x] TensorRT accelerated training is verified (accelerate inference on teacher network with TRT)
- [x] Implemented Soft Target Loss by [Hinton et. al.](https://arxiv.org/abs/1503.02531)
- [x] Implemented Hard Label Distillation by [Touvron et. al.](https://arxiv.org/abs/2012.12877)

# ToDo
- [ ] Improve TRT inference and training by transfering input only once.
- [ ] Benchmark dynamic shapes on TRT
- [ ] Better documentation
- [ ] Make PyPi package