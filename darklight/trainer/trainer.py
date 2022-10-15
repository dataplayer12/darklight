import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import os
import pdb
import sys
from ..losses.losses import SoftLabelsDistillationLoss, HardLabelDistillationLoss 
from ..trtengine.clsengine import TRTClassifier
from .timer import Timer
try:
	from apex import amp
	has_amp=True
except ImportError:
	print('AMP not found, mixed precision training not available')
	has_amp=False

class StudentTrainer(object):
	def __init__(self, net, dm, teacher_onnx=None, opt_params=None):
		'''
		net: (nn.Module) student net to train
		
		dm: data manager class as defined in ImageNetManager
		
		teacher_onnx: (str) path of onnx file for teacher network,
			use utils.exportonnx to export onnx file
			If teacher path is None, dark knowledge is not used
		
		opt_params: (dict) 
		{
			'optimizer': optimizer caller,
			'okwargs': keyword arguments to pass to optimizer constructor
			'scheduler': scheduler caller,
			'skwargs': keyword arguments to pass to init scheduler,
			'amplevel': None or 'O1' or 'O2'. If None, amp is not used
		}

		If opt_params is None, AdamW is used with a constant learning rate of 1e-4
		'''
		self.use_dark_knowledge=True if teacher_onnx is not None else False
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.net=net.to(self.device)
		self.dm=dm
		self.writer=SummaryWriter()
		self.criterion=SoftLabelsDistillationLoss(10) if self.use_dark_knowledge else nn.CrossEntropyLoss()

		if opt_params is None:
			self.optimizer=optim.AdamW(self.net.parameters(), lr=1e-4, weight_decay=0.05)
			self.has_scheduler= False #scheduler is not None
			self.amplevel=None
		else:
			self.optimizer=opt_params['optimizer'](self.net.parameters(), **opt_params['okwargs'])
			self.scheduler=opt_params['scheduler'](self.optimizer, **opt_params['skwargs'])
			self.amplevel=opt_params['amplevel']
			self.has_scheduler= True
			
		self.trtengine=None
		self.savepath=None
		self.get_accuracy=lambda p,y: (torch.argmax(p, dim=1) == y).to(torch.float).mean().item()

		if self.use_dark_knowledge:
			self.stream=torch.cuda.Stream() #support multi-GPU

			self.trtengine=TRTClassifier(
			teacher_onnx,
			nclasses=1000,
			insize=self.dm.size,
			imgchannels=3,
			maxworkspace=(1<<27),
			precision='FP16',
			device='GPU',
			max_batch_size=self.dm.bsize
			)

	def resume_training(self, loadpath, epochs, save, rstep=0):
		'''
		loadpath: (str) checkpoint path to resume from
		epochs: (int) number of epochs to train
		save: (str) savepath to format
		rstep: (int) step number to resume from (reflected in tensorboard logs)
		'''
		self.net.load_state_dict(torch.load(loadpath))
		self.train(epochs, save, rstep)

	def evaluate_model(self, step, verbose=False):
		'''
		step: (int) step to write in tensorboard log
		verbose: (bool) whether or not to print performance statistics
		'''
		self.net.eval()
		valoss=[]
		vaacc=[]
		with torch.no_grad():
			
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

		if has_amp and self.amplevel is not None:
			self.net, self.optimizer = amp.initialize(self.net, self.optimizer,
										opt_level=self.amplevel, enabled=True)

		self.net=nn.DataParallel(self.net) #support multi-GPU

		step=rstep
		
		nbatches=len(train_loader)
		timer=Timer()

		for epoch in range(epochs):
			timer.reset()
			for ix, (x,y) in enumerate(train_loader):
				self.optimizer.zero_grad()

				if self.use_dark_knowledge:
					with torch.cuda.stream(self.stream):
						#support multi-GPU
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

				if has_amp and self.amplevel is not None:
					with amp.scale_loss(loss, self.optimizer) as scaled_loss:
						scaled_loss.backward()
				else:
					loss.backward()

				self.optimizer.step()

				if self.has_scheduler:
					self.scheduler.step(epoch + ix/nbatches)

				acc=self.get_accuracy(pred, y)
				step+=1
				self.writer.add_scalar('Training Accuracy', acc, step)

				if step%eval_interval==0:
					print('Step: {}, Training throughput: {:.2f} im/s, loss= {:.5f}'.format( 
						step, (ix*self.dm.bsize)/timer.tick(), loss.item()))
					timer.pause()
					self.evaluate_model(step)
					self.net.train()
					timer.unpause()

			if self.use_dark_knowledge:
				oldt=self.criterion.temperature.item()
				newt= 1+0.9*(oldt-1)
				self.criterion.temperature= torch.tensor(newt)

			self.save(epoch)
			print('Time taken for last epoch = {:.3f}'.format(timer.tick(full=True)))

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
		trainer=StudentTrainer(net, dm)
		print('Created trainer')
		
		trainer.resume_training('mbv2_dk_try2/mbv2_dk_in1k_14.pth', 45, 'mbv2_dk_in1k_{}.pth', sstep=750000)
		#trainer.net.load_state_dict(torch.load('mbv2_dk_try2/mbv2_dk_in1k_14.pth'))
		#trainer.evaluate_model(step=75000, verbose=True)