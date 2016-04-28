# @OpService ops
# @DatasetIOService ds
# @OUTPUT ImgPlus image1
# @OUTPUT ImgPlus image2
# @OUTPUT ImgPlus image3
# @OUTPUT ImgPlus image4
# @OUTPUT ImgPlus image5

from net.imglib2 import Point

from net.imglib2.algorithm.region.hypersphere import HyperSphere

from ij import IJ

from mpicbg.imglib.interpolation.nearestneighbor import NearestNeighborInterpolatorFactory;

import random
import math

preys = []
classification = []
kernels = []

#load relevant images
for i in range(1,4):
	if (i == 1):
		curr_image = ds.open("/Users/test/Desktop/Git/Predator_vs_Prey/prey_images/grass_greyscale.jpg").getImgPlus()
	else:
		curr_image = ds.open("/Users/test/Desktop/Git/Predator_vs_Prey/prey_images/generation0000%d_max_prey.png" % (i * 3)).getImgPlus()
	
	#normalize image
#	curr_image = ops.convert().float32(curr_image)
#	cursor_image=curr_image.cursor()
#	mean = ops.stats().mean(curr_image)
#	std_dev = ops.stats().stdDev(curr_image)
#	
#	while ( cursor_image.hasNext()):
#		cursor_image.fwd()
#		cursor_image.get().set( (cursor_image.get().get() - mean.get())/ std_dev.get() )
		
	preys.append(curr_image)
	if (i == 1):
		curr_image = ds.open("/Users/test/Desktop/Git/Predator_vs_Prey/prey_images/grass_greyscale.jpg").getImgPlus()
		cursor_curr=curr_image.cursor()
	
		while ( cursor_curr.hasNext()):
			cursor_curr.fwd()
			cursor_curr.get().set(0)
		classification.append(curr_image)
	else:
		curr_image = ds.open("/Users/test/Desktop/Git/Predator_vs_Prey/prey_images/classification.png").getImgPlus()
#		cursor_curr=curr_image.cursor()
#	
#		while ( cursor_curr.hasNext()):
#			cursor_curr.fwd()
#			if (cursor_curr.get().get() == 0):
#				cursor_curr.get().set(-1)
#			else:
#				cursor_curr.get().set(1)
		classification.append(curr_image)
		
xSize = 250
ySize = 250
kernel_init=ops.create().kernelGauss([16, 16])
score1=0
for a in range(5):
	kernel_mod=ops.copy().img(kernel_init)		
	for i in range(100): 
			
		cursor_kernel_mod=kernel_mod.cursor()
		
		while ( cursor_kernel_mod.hasNext()):
				cursor_kernel_mod.fwd()
				cursor_kernel_mod.get().set( cursor_kernel_mod.get().get() + random.uniform(-.001, 0.001))
		
		threshold=95
		score2=0
		for b in range(3):
			convolved2=ops.filter().convolve(preys[b], kernel_mod)
			
			cursor_class=classification[b].cursor()
			cursor_convolve2=convolved2.cursor()
			
			# make function
			while ( cursor_convolve2.hasNext()):
				cursor_convolve2.fwd()
				cursor_class.fwd()
				if (cursor_convolve2.get().get() > threshold):
						cursor_convolve2.get().set( 1 )
				else:
						cursor_convolve2.get().set( 0 )	
						
				if ((cursor_convolve2.get().get() > 0 and cursor_class.get().get()) > 0 or (cursor_convolve2.get().get() <= 0 and cursor_class.get().get() <= 0)):
						score2 += 1	
			
		if (score2 > score1):
				kernel_init = ops.copy().img(kernel_mod)
				score1 = score2
		else:
				kernel_mod = ops.copy().img(kernel_init)	
	kernels.append(kernel_init)
	
	for i in range(3):
		convolved1=ops.filter().convolve(preys[i], kernel_init)
		preys[i] = convolved1
		cursor_convolve1=convolved1.cursor()
		cursor_class=classification[i].cursor()
		
		# make function
		while ( cursor_convolve1.hasNext()):
				cursor_convolve1.fwd()
				cursor_class.fwd()
				if (cursor_convolve1.get().get() > threshold):
						cursor_convolve1.get().set( 1 )
				else:
						cursor_convolve1.get().set( 0 )
							
				if ((cursor_convolve1.get().get() > 0 and cursor_class.get().get()) > 0 or (cursor_convolve1.get().get() <= 0 and cursor_class.get().get() <= 0)):
						cursor_convolve1.get().set( 1 )
				else:
						cursor_convolve1.get().set( 0 )
		classification[i] = convolved1
	print("iteration!")

curr_image = ds.open("/Users/test/Desktop/Git/Predator_vs_Prey/prey_images/grass_greyscale.jpg").getImgPlus()
for i in range(len(kernels)):
	curr_image = ops.filter().convolve(curr_image, kernels[i])

cursor_image=curr_image.cursor()
threshold = 95

while ( cursor_image.hasNext()):
		cursor_image.fwd()
		if (cursor_image.get().get() > threshold):
				cursor_image.get().set( 1 )
		else:
				cursor_image.get().set( 0 )

curr_image2 = ds.open("/Users/test/Desktop/Git/Predator_vs_Prey/prey_images/generation00012_max_prey.png").getImgPlus()
for i in range(len(kernels)):
	curr_image2 = ops.filter().convolve(curr_image2, kernels[i])
				
image1=ops.create().imgPlus(classification[0])
image1.setName("image1")

image2=ops.create().imgPlus(classification[1])
image2.setName("image2")

image3=ops.create().imgPlus(classification[2])
image3.setName("image3")

image4=ops.create().imgPlus(curr_image)
image4.setName("image4")

image5=ops.create().imgPlus(curr_image2)
image5.setName("image5")
