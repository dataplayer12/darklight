# Project DarkLight
Efficient Knowledge Distillation in neural networks using TensorRT inference on teacher network

# Background
Knowledge Distillation (KD) refers to the practice of using the outputs of a large teacher network train a (usually) smaller student network. This project leverages TensorRT to accelerate this process. It is common practice in KD, especially [dark knowledge](https://arxiv.org/abs/1503.02531) type techniques to pre-compute the logits from the teacher network and save them to disk. For training the student network, the pre-computed logits are used as is to teach the student. This saves GPU resources as one does not need to load the large teacher network to GPU memory during training.

# Problem
In [A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237), Beyer et. al. introduce the function matching approach for distilling the knowledge in a neural network. In this approach, rather than pre-computing the outputs from the teacher network, they are computed on the fly during training on the exact same input as seen by the student. However, this requires that the teacher model must share the GPU memory and compute resources and leads to the following question:

How to achieve the best teacher inference and student training performance on a GPU?

# Solution
- Use TensorRT to set up an inference engine and perform blazing fast inference
- Use logits from TensorRT inference to train the student network.
- Note that TensorRT works only on NVIDIA GPUs. AMD, Intel GPUs or TPUs are not supported.

# Environment

- Install with `pip`
DarkLight can now be installed via PyPi (pip) with
```Shell
pip install darklight
```

However, this will not install TensorRT, which can be installed with either of these methods.

- Recommended method
This project uses pytorch CUDA, tensorrt>=8.0, opencv and pycuda. The recommended way to get all these is to use an NGC docker container with a recent version of PyTorch.

```Shell
sudo docker run -it --ipc=host --net=host --gpus all nvcr.io/nvidia/pytorch:22.08-py3 /bin/bash
#if you want to load an external disk to the container, use the --volume switch

#Once the container is up and running, install pycuda
pip install pycuda darklight
```

- Custom env

If you want to use your own environment with PyTorch, you need to get TensorRT and pycuda.

Follow the [official guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip) to download TensorRT deb file and install it with the script provided in this repo. Finally install pycuda
```Shell
git clone https://github.com/dataplayer12/darklight.git
cd darklight
bash install_trt.sh
# if needed modify the version of deb file in the script before running.
# This script will also install pycuda
# this might fail for a number of reasons which is why NGC container is recommended
```

# Status
- [x] Currently supports image classification KD.
- [x] The core TensorRT functionality works well (can also be used for pure inference) 
- [x] TensorRT accelerated training is verified (accelerate inference on teacher network with TRT)
- [x] Implemented Soft Target Loss by [Hinton et. al.](https://arxiv.org/abs/1503.02531)
- [x] Implemented Hard Label Distillation by [Touvron et. al.](https://arxiv.org/abs/2012.12877)

# Immediate ToDos
- [ ] Improve TRT inference and training by transfering input only once.
- [ ] Benchmark dynamic shapes on TRT
- [ ] Benchmark PyTorch v/s TensorRT inference speed/memory
- [ ] Better documentation
- [ ] Better unit tests
- [x] Make PyPi package

# Roadmap

- This project will support KD on semantic segmentation
- KD support for object detection is planned.
- Teacher inference with other backends (OpenVINO, [MIVisionX](https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/), OpenCV DNN module) planned but are not a high priority.