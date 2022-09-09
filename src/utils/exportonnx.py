import torch
import os
import torchvision.models as models
import sys


BATCHSIZE=128 if len(sys.argv)==1 else int(sys.argv[1])

print('Using batch size: ', BATCHSIZE)

net=models.resnet152(pretrained=True)

x=torch.randn(BATCHSIZE,3,224,224)

inputn=['img']
outputn=['probs']
dynamic_axes = {'img': {0: 'bsize'}, 'probs': {0: 'bsize'}}

torch.onnx.export(net, x, 'resnet152_bs{}.onnx'.format(BATCHSIZE),input_names=inputn, output_names=outputn, dynamic_axes=dynamic_axes)

# It is not enough to use dynamic_axes in onnx. If you want TRT engine 
# to support all sizes from 1...BATCHSIZE, please set use_dynamic_shapes
# when constructing TRTClassifier to True