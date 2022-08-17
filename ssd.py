'''
	* Source
	https://www.kaggle.com/code/sdeagggg/ssd300-with-pytorch/notebook
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
from torchvision import models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from tqdm import tqdm
import time
import os
import json
import re
import random
from math import sqrt
from datetime import datetime

from lib.archives import *

from bs4 import BeautifulSoup as bs4

from random import randint

import warnings
warnings.filterwarnings('ignore')

dims = (300, 300)
'''
	# show sample
'''
DF_PATH = "./input/datasets_20220815.csv"
XML_PATH = "./input/annotations/Annotation/1002173/1002173_27480_23360_1024"

IMG_PATH = "./input/images/Images/1002173/1002173_27480_23360_1024.png"

def show_sample():
	# SHOW SAMPLE
	df = pd.read_csv(DF_PATH)

	xml = open(XML_PATH, 'r')
	xml = bs4(xml, "xml")

	xmin = int(xml.find("xmin").string)
	xmax = int(xml.find("xmax").string)
	ymin = int(xml.find("ymin").string)
	ymax = int(xml.find("ymax").string)

	img = Image.open(IMG_PATH)

	origin_img = img.copy()
	draw = ImageDraw.Draw(origin_img)
	draw.rectangle(xy=[(xmin,ymin), (xmax,ymax)], outline="red", width=5)

	plt.imshow(origin_img)
	plt.axis("off")
	plt.show()
	plt.close()

	old_dims = torch.FloatTensor([img.width, img.height, img.width, img.height]).unsqueeze(0)
	box = torch.FloatTensor([xmin, ymin, xmax, ymax])
	new_box = box / old_dims
	new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
	new_box = new_box * new_dims

	print("old dims:", old_dims)
	print("new box:", new_box)

	resize_img = transforms.Resize(dims)(img)
	draw = ImageDraw.Draw(resize_img)
	draw.rectangle(
		xy=[tuple(new_box.tolist()[0])[:2],
			tuple(new_box.tolist()[0])[2:]],
		outline="red",
		width=2
	)
	plt.imshow(resize_img)
	plt.axis("off")
	plt.show()
	plt.close()

	print(torch.FloatTensor([xmin, ymin, xmax, ymax]))

'''
	# config
'''
# Datasets
label_map = {
	'inf': 1,
	'base': 2,
	'sur': 3,
	'nuclear': 4,
	'int': 5,
	'test': 6,
	'xcell': 7
}

rev_label_map = {
	0: 'background',
	1: 'inf',
	2: 'base',
	3: 'sur',
	4: 'nuclear',
	5: 'int',
	6: 'test',
	7: 'xcell',
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 5
LR = 1e-3
BS = 8
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
print_feq = 100
IMG_ROOT = "./input/images/Images/"
XML_ROOT = "./input/annotations/Annotation"
all_img_folder = os.listdir(IMG_ROOT)

model = ''
def train():
	all_img_name = []
	for img_folder in all_img_folder:
		img_folder_path = "%s/%s" % (IMG_ROOT, img_folder)
		all_img_name += list(map(lambda x: img_folder + '/'+ x, os.listdir(img_folder_path)))
	i = 0
	while i < len(all_img_name):
		path = "%s/%s" % (XML_ROOT, all_img_name[i].replace(".png", ''))
		if not os.path.exists(path):
			all_img_name.remove(all_img_name[i])
			continue
		i += 1

	tsfm = transforms.Compose([
		transforms.Resize([300, 300]),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	])

	model = SSD().to(device)
	criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=momentum, weight_decay=weight_decay)

	train_ds = SSDDateset(all_img_name[:int(len(all_img_name)*0.8)], transform=tsfm)
	train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=train_ds.collate_fn)

	valid_ds = SSDDateset(all_img_name[int(len(all_img_name)*0.8):int(len(all_img_name)*0.9)], transform=tsfm)
	valid_dl = DataLoader(valid_ds, batch_size=BS, shuffle=True, collate_fn=valid_ds.collate_fn)


	for epoch in range(1, EPOCH+1):
		model.train()
		train_loss = []
		for step, (img, boxes, labels) in enumerate(train_dl):
			time_1 = time.time()
			img = img.cuda()
	#		 box = torch.cat(box)
			boxes = [box.cuda() for box in boxes]
	#		 label = torch.cat(label)
			labels = [label.cuda() for label in labels]

			pred_loc, pred_sco = model(img)

			loss = criterion(pred_loc, pred_sco, boxes, labels)

			 # Backward prop.
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
	#		 losses.update(loss.item(), images.size(0))
			train_loss.append(loss.item())
			if step % print_feq == 0:
				print('epoch:', epoch,
					  '\tstep:', step+1, '/', len(train_dl) + 1,
					  '\ttrain loss:', '{:.4f}'.format(loss.item()),
					  '\ttime:', '{:.4f}'.format((time.time()-time_1)*print_feq), 's')

		model.eval();
		valid_loss = []
		for step, (img, boxes, labels) in enumerate(tqdm(valid_dl)):
			img = img.cuda()
			boxes = [box.cuda() for box in boxes]
			labels = [label.cuda() for label in labels]
			pred_loc, pred_sco = model(img)
			loss = criterion(pred_loc, pred_sco, boxes, labels)
			valid_loss.append(loss.item())

		print('epoch:', epoch, '/', EPOCH+1,
				'\ttrain loss:', '{:.4f}'.format(np.mean(train_loss)),
				'\tvalid loss:', '{:.4f}'.format(np.mean(valid_loss)))

	date = datetime.now()
	date = datetime.strftime(date, "%Y%M%d")
	model_path = "./models/SSD_%s.pth" % date
	torch.save(model.state_dict(), model_path)
	del train_ds, train_dl, valid_ds, valid_dl

def test(n=5):
	model.eval()
	d = []
	for i in range(n):
		origin_img = Image.open(os.path.join(IMG_ROOT, all_img_name[-1*randint(1,25)]).convert('RGB'))
		img = tsfm(origin_img)

		img = img.cuda()
		predicted_locs, predicted_scores = model(img.unsqueeze(0))
		det_boxes, det_labels, det_scores = model.detect_objects(
			predicted_locs, predicted_scores, min_score=0.2,
			max_overlap=0.5, top_k=200
		)
		det_boxes = det_boxes[0].to('cpu')

		origin_dims = torch.FloatTensor([origin_img.width, origin_img.height, origin_img.width, origin_img.height]).unsqueeze(0)
		det_boxes = det_boxes * origin_dims

		annotated_image = origin_img
		draw = ImageDraw.Draw(annotated_image)

		box_location = det_boxes[0].tolist()
		draw.rectangle(xy=box_location, outline='red')
		draw.rectangle(xy=list(map(lambda x:x+1, box_location)), outline='red')
		d.append(annotated_image)
	return d

def test_exe():
	for i in test(n=10):
		plt.figure(figsize=(5, 5))
		plt.imshow(i)
		plt.axis("off")
		plt.show()
		plt.close()

if __name__ == "__main__":
	print("SSD Train-Test")
	# show sample
	# show_sample()

	# train
	# train()

	# test
	# test_exe()
