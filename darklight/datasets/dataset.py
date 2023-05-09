from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch
from .classes import IMAGENET2012_CLASSES
import os
from PIL import Image
import glob
import pycocotools

class ImageNet1k(Dataset):
	def __init__(self, root, dstype, size=[256, 256], extension="JPEG"):
		"""
		root: root directory of imagenet dataset
		dstype: one of train, val, test or wild (if you dont want to use labels)
		size: resize images to this shape
		"""
		super(ImageNet1k, self).__init__()

		patterns={'train': f"{root}/train_images/*.{extension}",
		'val': f"{root}/val_images/*.{extension}",
		'test': f"{root}/test_images/*.{extension}",
		'wild': f"{root}/train_images/*.{extension}"
		}

		self.image_paths=glob.glob(patterns[dstype])
		self.image_names=[f[f.rfind('/')+1:] for f in self.image_paths]
		
		if dstype == 'test' or dstype == 'wild':
			self.labels = None
		else:
			self.label_synset=[f.split('.')[0].split('_')[-1] for f in self.image_names]
			self.imagenet_synset, self.imagenet_descs=[], []

			for k,v in IMAGENET2012_CLASSES.items():
				self.imagenet_synset.append(k)
				self.imagenet_descs.append(v)

			self.labels=[self.imagenet_synset.index(l) for l in self.label_synset]

			assert len(self.imagenet_synset)==1000, 'Bug or not ImageNet1k'

		self.nclasses=1000
		self.inputsize=size

		self.transforms=self.random_transforms()

	def random_transforms(self):
		normalize_transform=T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		#define normalization transform with which the torchvision models
		#were trained

		affine=T.RandomAffine(degrees=5, translate=(0.05, 0.05))
		hflip =T.RandomHorizontalFlip(p=0.7)
		vflip =T.RandomVerticalFlip(p=0.7)
		
		blur=T.GaussianBlur(7) #kernel size 5x5

		rt1=T.Compose([T.Resize(self.inputsize), affine, T.ToTensor(), normalize_transform])
		rt2=T.Compose([T.Resize(self.inputsize), hflip, T.ToTensor(), normalize_transform])
		rt3=T.Compose([T.Resize(self.inputsize), vflip, T.ToTensor(), normalize_transform])
		rt4=T.Compose([T.Resize(self.inputsize), blur, T.ToTensor(), normalize_transform])

		return [rt1, rt2, rt3, rt4]

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		imgpath=self.image_paths[index]
		img=Image.open(imgpath).convert('RGB') 
		#some images are grayscale and need to be converted into RGB
	
		img_tensor=self.transforms[torch.randint(0,4,[1,1]).item()](img)

		if self.labels is None:
			return img_tensor
		else:
			label=self.labels[index]
			return img_tensor, label

class ImageNetManager(object):
	'''
	Creates and manages train/val/test datasets, dataloaders etc.
	Use of in-the-wild (unlabelled) datasets is supported for training
	but not for evaluation. Evaluation must always be imagenet like dataset
	'''
	def __init__(self, root, size=[224, 224], bsize=32, is_wild = False, num_workers=8, extension='JPEG'):
		self.root=root
		self.size=size
		self.bsize=bsize
		self.is_train_wild = is_wild
		self.num_workers = num_workers
		self.train_image_extension = extension
		self.train_loader, self.valid_loader= self.get_train_val_iters()

	def get_train_val_iters(self):
		return self.get_train_iterator(), self.get_valid_iterator()

	def get_train_iterator(self):
		dstype = 'wild' if self.is_train_wild else 'train'

		tdata=ImageNet1k(self.root, dstype, self.size, self.train_image_extension)
		tloader=DataLoader(tdata, self.bsize, shuffle=True, num_workers=self.num_workers, prefetch_factor=16, drop_last=True)
		return tloader

	def get_valid_iterator(self):
		vdata=ImageNet1k(self.root, 'val', self.size)
		vloader=DataLoader(vdata, self.bsize, shuffle=True, num_workers=self.num_workers//2, prefetch_factor=16)
		return vloader

	def get_test_iterator(self):
		tedata=ImageNet1k(self.root, 'test', self.size)
		teloader=DataLoader(tedata, self.bsize, shuffle=True, num_workers=self.num_workers//2)
		return teloader

class SegmentAnything(Dataset):
	def __init__(self, root):
		super(SegmentAnything, self).__init__()

		img_pattern=f"{root}/images/*"
		label_pattern=f"{root}/labels/*"

		self.image_paths=glob.glob(img_pattern)
		self.image_names=[f[f.rfind('/')+1:] for f in self.image_paths]
		
		self.label_paths=glob.glob(label_pattern)
		self.label_names=[f[f.rfind('/')+1:] for f in self.label_paths]

		self.input_transforms = self.random_transforms()
		self.label_transforms

	def __len__(self):
		return len(self.image_paths)
	
	def random_transforms(self):
		normalize_transform=T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

		affine=T.RandomAffine(degrees=5, translate=(0.05, 0.05))
		hflip =T.RandomHorizontalFlip(p=0.7)
		vflip =T.RandomVerticalFlip(p=0.7)
		
		blur=T.GaussianBlur(7) #kernel size 5x5

		rt1=T.Compose([affine, T.ToTensor(), normalize_transform])
		rt2=T.Compose([hflip, T.ToTensor(), normalize_transform])
		rt3=T.Compose([vflip, T.ToTensor(), normalize_transform])
		rt4=T.Compose([blur, T.ToTensor(), normalize_transform])

		return [rt1, rt2, rt3, rt4]

	def read_label(self, index):


	def __getitem__(self, index):
		imgpath=self.image_paths[index]
		img=Image.open(imgpath)

		rindex = torch.randint(0,4,[1,1]).item()
		img_tensor=self.input_transforms[rindex](img)
		label = self.read_label(index)
		label = self.transform_label(label, rindex)

		if self.labels is None:
			return img_tensor
		else:
			return img_tensor, label

if __name__=="__main__":
	pass
	im=ImageNetManager('/sfnvme/imagenet')
	t, v=im.get_train_val_iters()
	for x,y in t:
		print(x.shape, y.shape)
		break
