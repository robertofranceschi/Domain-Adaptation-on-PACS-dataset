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
		# use (0.5 0.5 0.5) (0.5 0.5 0.5) as mean and std
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()
		# raise RuntimeError("You should pass mean and std to 'imgshow' method")
	else : 
		# use custom mean and std computed on the images
		mean = np.array(mean)
		std = np.array(std)
		for i in range(3): 
			img[i] = img[i]*std[i] + mean[i] # unnormalize

		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()
	return

def plotImageDistribution(data1, data2, data3, data4, dataset_names, show=False):
  # concatenate datasets
  data = np.concatenate( (data1, data2, data3, data4) )
  # count element per class
  unique, counts = np.unique(data, return_counts=True)
  # for each domain
  unique, counts1 = np.unique(data1, return_counts=True)
  unique, counts2 = np.unique(data2, return_counts=True)
  unique, counts3 = np.unique(data3, return_counts=True)
  unique, counts4 = np.unique(data4, return_counts=True)

  if show: 
    print('Total images:', np.sum(counts))
    print('Number of classes:', len(unique))
    print('Classes:', unique)
    print('Total images per class:', counts)
    print('Mean images per class:', counts.mean())
    print('Std images per class:', counts.std())

  print()
  print('Element per class for each domain:')
  for name,count in zip(dataset_names,[counts1,counts2,counts3,counts4]) : 
    print(f'{name}_dataset: {count}')

  fig, ax = plt.subplots(figsize=(10,7))

  width=0.18

  #plt.bar(unique, counts, width=width, color=color)
  plt.bar(unique-2*(width)+(width/2), counts1, width=width, color='#FF8F77', linewidth=0.5, label='Photo')
  plt.bar(unique-(width/2), counts2, width=width, color='#FFDF77', linewidth=0.5, label='Art paintings')
  plt.bar(unique+(width/2), counts3, width=width, color='#8DF475', linewidth=0.5, label='Cartoon')
  plt.bar(unique+2*(width)-(width/2), counts4, width=width, color='#77DCFF', linewidth=0.5, label='Sketch')

  ax.set_xticks(unique)
  classes = ['Dog', 'Elephant', 'Giraffe', 'Guitar', 'Horse', 'House', 'Person']
  ax.set_xticklabels(classes)

  plt.grid(alpha=0.2, axis='y')

  plt.legend()
  plt.show()
  plt.savefig('distribution.png', dpi = 250)

def beep():
	# Play an audio beep. Any audio URL will do.
	output.eval_js('new Audio("https://upload.wikimedia.org/wikipedia/commons/0/05/Beep-09.ogg").play()')
