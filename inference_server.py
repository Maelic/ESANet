# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
from glob import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from socket import *
import json
from struct import unpack, pack
import base64
import time

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.prepare_data import prepare_data


def _load_img(fp):
	img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
	if img.ndim == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img

def base64_to_cv2(encoded_data):
	"""
	Convert an image in base64 format to cv2 format"

	:param encoded_data:
	:return:
	"""
	nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	return img

if __name__ == '__main__':
	# parser = argparse.ArgumentParser(description='Launch the instancee segmentation server.')
	# parser.add_argument("--server_ip", type=str, default="127.0.0.1", help="Server IP address. Default localhost '127.0.0.1'.")	
	# parser.add_argument("--server_port", type=int, default='9988', help="Server port address. Default '9988'.")	

	# args = parser.parse_args()

	CLASS_NAMES_40 = ['void',
					  'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
					  'table', 'door', 'window', 'bookshelf', 'picture',
					  'counter', 'blinds', 'desk', 'shelves', 'curtain',
					  'dresser', 'pillow', 'mirror', 'floor mat', 'clothes',
					  'ceiling', 'books', 'refridgerator', 'television',
					  'paper', 'towel', 'shower curtain', 'box', 'whiteboard',
					  'person', 'night stand', 'toilet', 'sink', 'lamp',
					  'bathtub', 'bag',
					  'otherstructure', 'otherfurniture', 'otherprop']

	# arguments
	parser = ArgumentParserRGBDSegmentation(
		description='Efficient RGBD Indoor Sematic Segmentation (Inference)',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.set_common_args()
	parser.add_argument('--ckpt_path', type=str,
						required=True,
						help='Path to the checkpoint of the trained model.')
	parser.add_argument('--depth_scale', type=float,
						default=1.0,
						help='Additional depth scaling factor to apply.')
	parser.add_argument('--server_ip', type=str,
						default="127.0.0.1",
						help='Server IP')
	parser.add_argument('--server_port', type=int,
						default=9988,
						help='Server port')
	args = parser.parse_args()

	# dataset
	args.pretrained_on_imagenet = False  # we are loading other weights anyway
	dataset, preprocessor = prepare_data(args, with_input_orig=True)
	n_classes = dataset.n_classes_without_void

	# model and checkpoint loading
	model, device = build_model(args, n_classes=n_classes)
	checkpoint = torch.load(args.ckpt_path,
							map_location=lambda storage, loc: storage)
	model.load_state_dict(checkpoint['state_dict'])
	print('Loaded checkpoint from {}'.format(args.ckpt_path))

	model.eval()
	model.to(device)

	# get samples
	basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
							'samples/test')
	rgb_filepaths = sorted(glob(os.path.join(basepath, 'image_rgb.png')))
	depth_filepaths = sorted(glob(os.path.join(basepath, 'image_depth.png')))
	assert args.modality == 'rgbd', "Only RGBD inference supported so far"
	assert len(rgb_filepaths) == len(depth_filepaths)
	filepaths = zip(rgb_filepaths, depth_filepaths)
	print(rgb_filepaths)
	socket = socket(AF_INET, SOCK_STREAM)
	socket.bind(("127.0.0.1", 9988))
	socket.listen(1)
	print("Listening")

	try:
		while True:
			(connection, addr) = socket.accept()
			try:
				connection.recv(1000)
				# (length,) = unpack('>Q', bs)
				# data = b''
				# while len(data) < length:
				# 	to_read = length - len(data)
				# 	data += connection.recv(
				# 		4096 if to_read > 4096 else to_read)

				# response = json.loads(data.decode('utf-8'))
				# im_rgb = base64_to_cv2(response['image'])
				# im_depth = base64_to_cv2(response['image_depth'])
				# cv2.imwrite(rgb_filepath, im_rgb)
				# cv2.imwrite(depth_filepath, im_depth)

				# load sample
				img_rgb = _load_img(rgb_filepaths[0])
				img_depth = _load_img(depth_filepaths[0])

				h, w, _ = img_rgb.shape

				# preprocess sample
				sample = preprocessor({'image': img_rgb, 'depth': img_depth})
				# add batch axis and copy to device
				image = sample['image'][None].to(device)
				depth = sample['depth'][None].to(device)
			
				# apply network
				pred = model(image, depth)

				pred = F.interpolate(pred, (h, w),
									 mode='bilinear', align_corners=False)

				pred = torch.argmax(pred, dim=1)

				pred = pred.cpu().numpy().squeeze().astype(np.uint8)
				detected_classes_names = []
				for prediction in pred:
					pred_pixel = list(set(prediction))
					for pixel in pred_pixel:
						if CLASS_NAMES_40[pixel] not in detected_classes_names:
							detected_classes_names.append(CLASS_NAMES_40[pixel])			
				#print(detected_classes_names)
				# show result
				pred_colored = dataset.color_label(pred, with_void=False)
				cv2.imshow('Demo', pred_colored)
				cv2.waitKey(3)

				result = detected_classes_names

				res_bytes = json.dumps(result).encode('utf-8') 

				connection.sendall(res_bytes)
			finally:
				connection.close()
	finally:
		self.socket.close()

			