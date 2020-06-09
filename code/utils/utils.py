<<<<<<< HEAD:code/utils.py
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# SHOW SOME RANDOM IMAGES
def show_random_images(dataset, n=5, mean=None, std=None):
	for i in range(n):
	    j = np.random.randint(0, len(dataset))
	    print('Label:',dataset[j][1])
	    imgshow(dataset[j][0], mean=mean, std=std)

	return

def imgshow(img, mean=None, std=None):
	if mean == None or std == None:
		raise RuntimeError("You should pass mean and std to 'imgshow' method")

	mean = np.array(mean)
	std = np.array(std)
	for i in range(3):
		img[i] = img[i]*std[i] + mean[i] # unnormalize

	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

	return

def beep():
	# Play an audio beep. Any audio URL will do.
	output.eval_js('new Audio("https://upload.wikimedia.org/wikipedia/commons/0/05/Beep-09.ogg").play()')
=======
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# SHOW SOME RANDOM IMAGES
def show_random_images(dataset, n=5, mean=None, std=None):
	for i in range(n):
	    j = np.random.randint(0, len(dataset))
	    print('Label: ',dataset[j][1])
	    imgshow(dataset[j][0], mean=mean, std=std)
	return

def imgshow(img, mean=None, std=None):
	# show images
	if mean == None or std == None:
		raise RuntimeError("You should pass mean and std to 'imgshow' method")

	mean = np.array(mean)
	std = np.array(std)
	# unnormalize 
	img = [img[i]*std[i] + mean[i] for i in range(3)] # 3 is the number of channel of each image
	
	# for i in range():
	# 	img[i] = img[i]*std[i] + mean[i] # unnormalize

	img = img.numpy()
	plt.imshow(np.transpose(img, (1, 2, 0)))
	plt.show()

	return

def beep():
	# Play an audio beep when called.
	output.eval_js('new Audio("https://upload.wikimedia.org/wikipedia/commons/0/05/Beep-09.ogg").play()')
>>>>>>> b2f28ef... upload from local repo:code/utils/utils.py
