import trtinfer
import sys
import torch
import torchvision.models as models
sys.path.insert(0, '../simple_vit/')

tvmodel=models.resnet152(pretrained=True)

clsr=trtinfer.TRTClassifier(
		'resnet152.onnx',
		nclasses=1000,
		insize=(224,224),
		imgchannels=3,
		maxworkspace=(1<<25), 
		precision='FP32', 
		device='GPU', 
		max_batch_size=1
		)

from dataset import ImageNetManager
dm=ImageNetManager('/sfnvme/imagenet/',size=[224,224] ,bsize=1)

it=iter(dm.train_loader)

x,y = next(it)
#x=x.cuda()
#y=y.cuda()

print(clsr.infer(x, benchmark=True, transfer=True))

ix=0
for x,y in it:
	#x=x.cuda()
	#y=y.cuda()
	clsr.infer(x, benchmark=True, transfer=True)
	trtout=clsr.output.argmax(axis=1)
	print(clsr.output)
	with torch.no_grad():
		tvout=tvmodel(x).numpy()
		print(tvout)
		tvout=tvout.argmax(axis=1)

	racc= (trtout==tvout).sum()
	aacc = (trtout == y.numpy()).sum()
	print(racc, aacc)
	ix+=1
	if ix==10:
		print(tvout.shape)
		break