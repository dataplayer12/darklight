import trtinfer
import sys
import torch
import torchvision.models as models
sys.path.insert(0, '../simple_vit/')

BATCHSIZE=128 if len(sys.argv)==1 else int(sys.argv[1])

tvmodel=models.resnet152(pretrained=True).eval()
calibrator=trtinfer.Calibrator('./val2017/', 100, input_size=(224,224))

clsr=trtinfer.TRTClassifier(
		'resnet152.onnx',
		nclasses=1000,
		insize=(224,224),
		imgchannels=3,
		maxworkspace=(1<<27), 
		precision='FP16',
		calibrator=calibrator,
		device='GPU', 
		max_batch_size=BATCHSIZE,
		use_dynamic_shapes=False
		)

from dataset import ImageNetManager
dm=ImageNetManager('/sfnvme/imagenet/',size=[224,224] ,bsize=BATCHSIZE)

it=iter(dm.train_loader)

x,y = next(it)
#x=x.cuda()
#y=y.cuda()

#print(clsr.infer(x, benchmark=True, transfer=True))

ix=0
for x,y in it:
	#x=x.cuda()
	#y=y.cuda()
	#sh=torch.cuda.current_stream()
	clsr.infer(x, benchmark=True, transfer=True)#, sh=sh)
	trtout=clsr.output.argmax(axis=1)
	#print('accuracy=',(trtout == y.numpy()).sum())
	with torch.no_grad():
		tvout=tvmodel(x).to('cpu')
		trtp=torch.nn.functional.softmax(torch.tensor(clsr.output), dim=1)
		tvp=torch.nn.functional.softmax(tvout, dim=1)
		diff=abs(trtp-tvp).sum(axis=1).mean().item()
		tvout=tvout.numpy().argmax(axis=1)

	racc= (trtout==tvout).sum()
	aacc = (trtout == y.numpy()).sum()
	
	print(racc, aacc, diff)
	ix+=1
	if ix==10:
		print(tvout.shape)
		break