import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import os
import pdb
import sys

try:
	from apex import amp
except ImportError:
	print('AMP not found')

class StudentTrainer(object):
	def __init__(self, net, dm, teacher_onnx=None):
		'''
		net: student net to train
		dm: data manager class as defined in ImageNetManager
		teacher_onnx: path of onnx file for tescher network,
		use utils.exportonnx to export onnx file
		If teacher path is None, dark knowledge is not use
		'''
		self.use_dark_knowledge=True if teacher_onnx is not None else False
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.net=net.to(self.device)
		self.dm=dm
		self.writer=SummaryWriter()
		self.criterion=SoftLabelsLoss(10) if self.use_dark_knowledge else nn.CrossEntropyLoss()
		self.optimizer=optim.AdamW(self.net.parameters(), lr=1e-5, weight_decay=0.05)
		self.scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
			self.optimizer,
			T_0=2,
			T_mult=2
			)
		self.trtengine=None
		self.savepath=None
		self.get_accuracy=lambda p,y: (torch.argmax(p, dim=1) == y).to(torch.float).mean().item()

		if self.use_dark_knowledge:
			self.trtengine=TRTClassifier(
			'resnet152_bs{}.onnx'.format(BATCH_SIZE),
			nclasses=1000,
			insize=(224,224),
			imgchannels=3,
			maxworkspace=(1<<27), 
			precision='FP16', 
			device='GPU', 
			max_batch_size=BATCH_SIZE
			)

	def resume_training(self, loadpath, epochs, save, sstep=0):
		self.net.load_state_dict(torch.load(loadpath))
		self.train(epochs, save, sstep)

	def evaluate_model(self, step, verbose=False):
		self.net.eval()
		valoss=[]
		vaacc=[]
		with torch.no_grad():
			pass
			for imgs, ys in self.dm.valid_loader:
				imgs=imgs.to(self.device)
				ys=ys.to(self.device)
				preds=self.net(imgs)
				vacc=self.get_accuracy(preds, ys)
				vloss=0 #self.criterion(preds, ys)
				
				valoss.append(vloss)
				vaacc.append(vacc)

		avgloss=np.mean(valoss)
		avgacc=np.mean(vaacc)
		self.writer.add_scalar('Validation Loss', avgloss, step)
		self.writer.add_scalar('Validation Accuracy', avgacc, step)
		if verbose:
			print('Validation loss= {:.3f}, validation accuracy= {:.2f}'.format(avgloss, 100*avgacc))

	def train(self, epochs, save, rstep=0):
		pass
		eval_interval=1000
		self.savepath=save
		
		train_loader = self.dm.train_loader #ignore test loader if any

		self.net.to(self.device).train()

		# self.net, self.optimizer = amp.initialize(self.net, self.optimizer,
		# 							opt_level='O2', enabled=True)

		step=rstep
		
		nbatches=len(train_loader)

		for epoch in range(epochs):
			estart=time.time()
			for ix, (x,y) in enumerate(train_loader):
				self.optimizer.zero_grad()

				if self.use_dark_knowledge:
					self.trtengine.infer(x, benchmark=False, transfer=True)
					#print('num_c=', (self.trtengine.output.argmax(axis=1)==y.numpy()).sum())
					yt=torch.tensor(self.trtengine.output)
					yt=yt.to(self.device)

				x=x.to(self.device) #transfer to GPU after trt inference
				y=y.to(self.device)
				
				pred = self.net(x)

				if self.use_dark_knowledge:
					loss = self.criterion(pred, yt, y)
				else:
					loss = self.criterion(pred, y)
				
				self.writer.add_scalar('Training Loss', loss.item(), step)

				# with amp.scale_loss(loss, self.optimizer) as scaled_loss:
				# 	scaled_loss.backward()

				loss.backward()

				self.optimizer.step()
				self.scheduler.step(epoch + ix/nbatches)
				acc=self.get_accuracy(pred, y)
				step+=1
				self.writer.add_scalar('Training Accuracy', acc, step)

				if step%eval_interval==0:
					ctime=time.time()
					print('Step: {}, Training throughput: {:.2f} im/s, loss= {:.5f}'.format( 
						step, (ix*self.dm.bsize)/(ctime-estart), loss.item()))
					self.evaluate_model(step)
					self.net.train()

			if self.use_dark_knowledge:
				oldt=self.criterion.temperature.item()
				newt= 1+0.9*(oldt-1)
				self.criterion.temperature= torch.tensor(newt)

			self.save(epoch)
			eend=time.time()
			print('Time taken for last epoch = {:.3f}'.format(eend-estart))

	def save(self, epoch):
		if self.savepath:
			path=self.savepath.format(epoch)
			torch.save(self.net.state_dict(), path)
			print(f'Saved model to {path}')

if __name__=="__main__":
	import timm
	from ..dataset.dataset import ImageNetManager
	from ..trtengine import TRTClassifier
	net=timm.create_model('mobilevit_s', pretrained=False)
	nparams=sum(p.numel() for p in net.parameters() if p.requires_grad)
	print(f'Created model with {nparams} parameters')

	#net.load_state_dict(torch.load('./mbv2/mbv2_in1k_24.pth'))
	#print('Loaded model checkpoint')

	dm=ImageNetManager('/sfnvme/imagenet/',size=[224,224] ,bsize=BATCH_SIZE)

	print('Created datamanager')

	stream=torch.cuda.Stream()

	with torch.cuda.stream(stream):
		trainer=CookTrainer(net, dm)
		print('Created trainer')
		
		trainer.resume_training('mbv2_dk_try2/mbv2_dk_in1k_14.pth', 45, 'mbv2_dk_in1k_{}.pth', sstep=750000)
		#trainer.net.load_state_dict(torch.load('mbv2_dk_try2/mbv2_dk_in1k_14.pth'))
		#trainer.evaluate_model(step=75000, verbose=True)