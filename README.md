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
- Either use a NGC docker container with a recent version of PyTorch, or
- Follow the [official guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip) to install TensorRT in your environment.

# Status
This is work in progress. Stay tuned...
