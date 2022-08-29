import torch
import os
import torchvision.models as models

net=models.resnet152(pretrained=True)

x=torch.randn(1,3,224,224)

inputn=['img']
outputn=['probs']
#dynamic_axes = {'img': {0: 'bsize'}, 'probs': {0: 'bsize'}}

torch.onnx.export(net, x, 'resnet152.onnx',input_names=inputn, output_names=outputn)#, dynamic_axes=dynamic_axes)