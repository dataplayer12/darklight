import torch.nn as nn

class SoftLabelsLoss(nn.Module):
	def __init__(self, temp=10, lamda=0.5):
		super(SoftLabelsLoss, self).__init__()
		self.register_buffer('temperature', torch.tensor(temp))
		self.register_buffer('lamda', torch.tensor(lamda))
		self.klloss=nn.KLDivLoss(reduction='batchmean')
		self.celoss=nn.CrossEntropyLoss(label_smoothing=0.1)

	def forward(self, model_output, dense_targets, class_targets):
		"""
		model_output: [bsize, nclasses] logit from student model
		dense_targets:[bsize, nclasses] logit from teacher model
		class_targets:[bsize] index of true class label
		"""
		model_probabilities=nn.functional.softmax(model_output/self.temperature, dim=1)
		target_probabilities=nn.functional.softmax(dense_targets/self.temperature, dim=1)
		kloss=self.klloss(torch.log(model_probabilities), target_probabilities)
		celoss=self.celoss(model_output, class_targets)
		loss= (1-self.lamda)*celoss+self.lamda*(self.temperature**2)*kloss
		return loss

class HardLabelDistillationLoss(nn.Module):
	def __init__(self, lamda=0.5, smoothing=0.1):
		super().__init__()
		self.register_buffer('lamda', torch.tensor(lamda))
		self.register_buffer('smoothing', torch.tensor(smoothing))

	def forward(self, model_output, teacher_output, true_label):
		teacher_preds=teacher_output.argmax(dim=1)

		lloss=nn.functional.cross_entropy(
			model_output, 
			true_label, 
			label_smoothing=self.smoothing
			)

		tloss=nn.functional.cross_entropy(
			model_output,
			teacher_preds,
			label_smoothing=self.smoothing
			)

		loss=self.lamda*lloss+(1-self.lamda)*tloss
		return loss