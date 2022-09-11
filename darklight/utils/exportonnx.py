import torch
import os
import torchvision.models as models
import sys

def exportonnx(net, outfile, bsize, hw=[224,224]):
    BATCHSIZE= bsize #128 if len(sys.argv)==1 else int(sys.argv[1])
    
    print('Using batch size: ', BATCHSIZE)

    #net=models.resnet152(pretrained=True)

    x=torch.randn(BATCHSIZE,3,hw[0],hw[1])

    inputn=['img']
    outputn=['probs']
    dynamic_axes = {'img': {0: 'bsize'}, 'probs': {0: 'bsize'}}

    torch.onnx.export(net, x, outfile, input_names=inputn, output_names=outputn, dynamic_axes=dynamic_axes)

    print(f"ONNX file successfully exported to {outfile}")

# It is not enough to use dynamic_axes in onnx. If you want TRT engine 
# to support all sizes from 1...BATCHSIZE, please set use_dynamic_shapes
# when constructing TRTClassifier to True
