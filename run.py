#!/usr/bin/env python
import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave, imread
import getopt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import math
import os
import PIL
import PIL.Image
import sys
from video_processing import frame_capture


##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 41) # requires at least pytorch version 0.4.1

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

Backward_tensorGrid = {}

def Backward(tensorInput, tensorFlow):
	if str(tensorFlow.size()) not in Backward_tensorGrid:
		tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
		tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

		Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
	# end

	tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		class Preprocess(torch.nn.Module):
			def __init__(self):
				super(Preprocess, self).__init__()
			# end

			def forward(self, tensorInput):
				tensorBlue = (tensorInput[:, 0:1, :, :] - 0.406) / 0.225
				tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
				tensorRed = (tensorInput[:, 2:3, :, :] - 0.485) / 0.229

				return torch.cat([ tensorRed, tensorGreen, tensorBlue ], 1)
			# end
		# end

		class Basic(torch.nn.Module):
			def __init__(self, intLevel):
				super(Basic, self).__init__()

				self.moduleBasic = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
				)
			# end

			def forward(self, tensorInput):
				return self.moduleBasic(tensorInput)
			# end
		# end

		self.modulePreprocess = Preprocess()

		self.moduleBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

		self.load_state_dict(torch.load('./network-chairs-final.pytorch'))
	# end

	def forward(self, tensorFirst, tensorSecond):
		tensorFirst = [ self.modulePreprocess(tensorFirst) ]
		tensorSecond = [ self.modulePreprocess(tensorSecond) ]

		for intLevel in range(5):
			if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
				tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2, count_include_pad=False))
				tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2, count_include_pad=False))
			# end
		# end

		tensorFlow = tensorFirst[0].new_zeros([ tensorFirst[0].size(0), 2, int(math.floor(tensorFirst[0].size(2) / 2.0)), int(math.floor(tensorFirst[0].size(3) / 2.0)) ])

		for intLevel in range(len(tensorFirst)):
			tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

			if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
			if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

			tensorFlow = self.moduleBasic[intLevel](torch.cat([ tensorFirst[intLevel], Backward(tensorInput=tensorSecond[intLevel], tensorFlow=tensorUpsampled), tensorUpsampled ], 1)) + tensorUpsampled
		# end

		return tensorFlow
	# end
# end

moduleNetwork = Network().cuda().eval()

##########################################################
def estimate(tensorFirst, tensorSecond):
	assert(tensorFirst.size(1) == tensorSecond.size(1))
	assert(tensorFirst.size(2) == tensorSecond.size(2))

	intWidth = tensorFirst.size(2)
	intHeight = tensorFirst.size(1)

#	assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
#	assert(intHeight == 416) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
	tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

	tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tensorFlow = torch.nn.functional.interpolate(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
	return tensorFlow[0, :, :, :].cpu()
# end

##########################################################
def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def compute_color(u, v):
	"""
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
	[h, w] = u.shape
	img = np.zeros([h, w, 3])
	nanIdx = np.isnan(u) | np.isnan(v)
	u[nanIdx] = 0
	v[nanIdx] = 0

	colorwheel = make_color_wheel()
	ncols = np.size(colorwheel, 0)

	rad = np.sqrt(u**2+v**2)

	a = np.arctan2(-v, -u) / np.pi

	fk = (a+1) / 2 * (ncols - 1) + 1

	k0 = np.floor(fk).astype(int)

	k1 = k0 + 1
	k1[k1 == ncols+1] = 1
	f = fk - k0

	for i in range(0, np.size(colorwheel,1)):
		tmp = colorwheel[:, i]
		col0 = tmp[k0-1] / 255
		col1 = tmp[k1-1] / 255
		col = (1-f) * col0 + f * col1

		idx = rad <= 1
		col[idx] = 1-rad[idx]*(1-col[idx])
		notidx = np.logical_not(idx)

		col[notidx] *= 0.75
		img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

	return img


def plot_optical_flows(video, cap):
	dir = video + "_frames/"
	heatmap_dir = video+"_heatmaps/"
	optical_flow_dir = video + "_optical_flow/"
	if not os.path.exists(heatmap_dir):
		os.mkdir(heatmap_dir)
	if not os.path.exists(optical_flow_dir):
		os.mkdir(optical_flow_dir)
	optical_flow = []
	for i in range(5,cap-5):
		tensorFirst = torch.FloatTensor(
			np.array(PIL.Image.open(dir+"frame" + str(i) + ".jpg"))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
			* (1.0 / 255.0))
		tensorSecond = torch.FloatTensor(
			np.array(PIL.Image.open(dir+"frame" + str(i+1) + ".jpg"))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (
			1.0 / 255.0))

		tensorOutput = estimate(tensorFirst, tensorSecond)
		tensorOutput = tensorOutput.numpy()
		np.save(optical_flow_dir + str(i), tensorOutput)
		#img = compute_color(tensorOutput[0], tensorOutput[1])
		#optical_flow.append(np.linalg.norm([np.sum(tensorOutput[0]), np.sum(tensorOutput[1])]))
	#	imsave(heatmap_dir + str(i) + ".png", img)
		if i % 100==0:
			print("finished comparing frame " + str(i) + " video " + video)

	# plt.close()
	# plt.plot(optical_flow)
	# np.save("numpy_"+video, optical_flow)
	# plt.savefig("flow_chart_" + video)
	# with open("log_" + video + ".txt", "w") as f:
	# 	f.write("The sum of optical flow is  " + str(sum(optical_flow)))
	# 	f.write("  There are " + str(cap) + " frames in total")



if __name__ == '__main__':

	# video_dir = "/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/videos/"
	# for root, dirs, files in os.walk(video_dir):
	# 	for file in files:
	# 		path = os.path.join(root, file)
	# 		video = file[:-4]
	# 		if os.path.exists("numpy_" + video+".npy"):
	# 			print("continue")
	# 			continue
	# 		count = frame_capture(path, video)
	# 		print("Finished capturing frames, there are ", count, " frames in total for ", video)
	# 		plot_optical_flows(video, count)

	video_names = ['dotsouter', 'gardens', 'glowingdance', 'helicoptercrash', 'minecraft', 'sculptures', 
					'sharks', 'ship', 'skyhouse', 'spacevisit', 'woodencoaster', 'cartoonroaster', 'dotsall', 'dunerovers', 
					'oceancoaster', 'snowplanet']
	for video_name in video_names:
		video_frames = "/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/original-frames/" + video_name + "_frames/"
		optical_flow_dir = "/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/original_optical_flow/" + video_name + "/"
		if not os.path.exists(video_frames):
    			continue
		if not os.path.exists(optical_flow_dir):
    			os.makedirs(optical_flow_dir)
		frame_count = len([name for name in os.listdir(video_frames) if os.path.isfile(os.path.join(video_frames, name))])
		print(frame_count) 
		tensorFirst = torch.FloatTensor(
			np.array(PIL.Image.open(video_frames + video_name + "00000.jpeg"))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
			* (1.0 / 255.0))
		for i in range(1, frame_count-1): 
			tensorSecond = torch.FloatTensor(
				np.array(PIL.Image.open(video_frames + video_name + "%05d" % i + ".jpeg"))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (
				1.0 / 255.0))
			tensorOutput = estimate(tensorFirst, tensorSecond)
			tensorOutput = tensorOutput.numpy()
			np.save(optical_flow_dir + str(i), tensorOutput)
			if i % 100==0:
				print("finished comparing frame " + str(i) + " video " + video_name)
			tensorFirst = tensorSecond

