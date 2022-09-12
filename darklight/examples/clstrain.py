'''
A minimal example demonstrating the use of resnet152 to train resnet18
'''
import darklight as dl
import torch
from torchvision import models

teacher=models.resnet152(pretrained=True) #substitue these with any other models you write
student= models.resnet18(pretrained=False)

dl.exportonnx(teacher, 'rn152.onnx', bsize=1, hw=[224,224])

del teacher #free up CPU or GPU memory used by teacher pytorch model

dm=dl.ImageNetManager('/sfnvme/imagenet/', size=[224,224], bsize=128)

opt_params={
	'optimizer': torch.optim.AdamW,
	'okwargs': {'lr': 1e-4, 'weight_decay':0.05},
	'scheduler':torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
	'skwargs': {'T_0':10,'T_mult':2},
	'amplevel': None
	}

stream=torch.cuda.Stream()

with torch.cuda.stream(stream):
	#TensorRT inference engine is constructed for the teacher from onnx file
	#CUDA stream scope ensures interoperability between pycuda, TensorRT and pytorch
	trainer=dl.StudentTrainer(student, dm, 'rn152.onnx', opt_params=opt_params)
	trainer.train(epochs=50, save='dltest_{}.pth')