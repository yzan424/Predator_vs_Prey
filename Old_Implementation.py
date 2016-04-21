# @OpService ops
# @DatasetIOService ds
# @OUTPUT ImgPlus image1
# @OUTPUT ImgPlus image2
# @OUTPUT ImgPlus image3

# preys is the original, convolved is convolved, and psf is kernel

#TODO: Make this into 2 dimensions, load pre-existing images of prey, wiki.imagej.net (develop -> scripting, jython scripting, look at templates tutorials)
from net.imglib2 import Point

from net.imglib2.algorithm.region.hypersphere import HyperSphere

from ij import IJ

from mpicbg.imglib.interpolation.nearestneighbor import NearestNeighborInterpolatorFactory;

import random

# create an empty image

preys = []
classification = []
kernels = []
for i in range(1,4):
#	preys.append(ops.create().imgPlus(ds.open("/Users/test/Desktop/prey_images/grass_greyscale.jpg")))
	preys.append(ops.create().imgPlus(ds.open("/Users/test/Desktop/prey_images/generation0000%d_max_prey.png" % (i * 3))))
	classification.append(ops.create().imgPlus(ds.open("/Users/test/Desktop/prey_images/classification.png")))

xSize = 250
ySize = 250

# use the randomAccess interface to place points in the image

# synthesizing an image 19 - 34
#randomAccess= preys.randomAccess()
#randomAccess.setPosition([xSize/2, ySize/2])
#randomAccess.get().setReal(255.0)
#
#randomAccess.setPosition([xSize/4, ySize/4])
#randomAccess.get().setReal(255.0)
#
#location = Point(preys.numDimensions())
#location.setPosition([3*xSize/4, 3*ySize/4])
#
#hyperSphere = HyperSphere(preys, location, 5)
#
#for value in hyperSphere:
#        value.setReal(16)

# preys.setName("preys")

# create psf using the gaussian kernel op (alternatively PSF could be an input to the script)
# convolved1=ops.filter().convolve(preys, psf1)
#psf2=ops.create().kernelGauss([5, 5])
#psf3=ops.create().kernelGauss([10, 10])

#calculate score convolved 1, old score
#within loop, calculate new score, if its greater, replace old kernel and old score
for a in range(5):
	kernel_init=ops.create().kernelGauss([16, 16])
	kernel_mod=ops.copy().img(kernel_init)
#	initial_convolution=ops.filter().convolve(preys[0], kernel_init)
				
	for i in range(100): 
			
		cursor_kernel_init=kernel_init.cursor()
		cursor_kernel_mod=kernel_mod.cursor()
		
		while ( cursor_kernel_mod.hasNext()):
				cursor_kernel_mod.fwd()
				cursor_kernel_mod.get().set( cursor_kernel_mod.get().get() + random.uniform(-.001, 0.001))
		
		threshold=95
		score1=0
		score2=0
		
		for b in range(3):
			convolved1=ops.filter().convolve(preys[b], kernel_init)	
			convolved2=ops.filter().convolve(preys[b], kernel_mod)
			
			cursor_class=classification[b].cursor()
			cursor_convolve1=convolved1.cursor()
			cursor_convolve2=convolved2.cursor()
			
			# make function
			while ( cursor_convolve1.hasNext()):
				cursor_convolve1.fwd()
				cursor_convolve2.fwd()
				cursor_class.fwd()
				if (cursor_convolve1.get().get() > threshold):
						cursor_convolve1.get().set( 1 )
				else:
						cursor_convolve1.get().set( 0 )
						
				if (cursor_convolve2.get().get() > threshold):
						cursor_convolve2.get().set( 1 )
				else:
						cursor_convolve2.get().set( 0 )	
							
				if ((cursor_convolve1.get().get() > 0 and cursor_class.get().get()) > 0 or (cursor_convolve1.get().get() <= 0 and cursor_class.get().get() <= 0)):
						score1 += 1
						
				if ((cursor_convolve2.get().get() > 0 and cursor_class.get().get()) > 0 or (cursor_convolve2.get().get() <= 0 and cursor_class.get().get() <= 0)):
						score2 += 1	
#			if (i == 0):
#					initial_black_white=ops.create().imgPlus(convolved1)
#					initial_black_white.setName("initial_black_white")
		#print(score1)
		#print(score2)
		if (score2 > score1):
				kernel_init = ops.copy().img(kernel_mod)
		else:
				kernel_mod = ops.copy().img(kernel_init)
					
#	final_convolution=ops.filter().convolve(preys[0], kernel_mod)
	kernels.append(kernel_init)
	
	for i in range(3):
		convolved1=ops.filter().convolve(preys[i], kernel_init)
		
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
						cursor_convolve1.get().set( 0 )
				else:
						cursor_convolve1.get().set( 1 )
		classification[i] = convolved1
	
# THE FOLLOWING CODE IS GARBAGE FROM TESTING EARLIER THAT HAS BEEN COMMENTED OUT			
		
# optimizing learning algorithm
# multiple inputs training simultaneously, which means training on more than one image
# generalization over background image
# 





## add two convolved images together
#composite1 = ops.math().add(convolved1, convolved2)
#
## feed composite as input into another convolution
#composite2=ops.filter().convolve(composite1, psf3)
## composite2=ops.image().scale( composite2, [0.1, 0.1], NearestNeighborInterpolatorFactory() )
#
## make convolved and composite an ImgPlus
#composite1=ops.create().imgPlus(composite1);
#composite1.setName("composite1")
#
#composite2=ops.create().imgPlus(composite2);
#composite2.setName("composite2")

# IJ.run(composite2, "Scale...", "x=0.1 y=0.1 interpolation=None average create")
#
#final_convolution=ops.create().imgPlus(final_convolution)
#final_convolution.setName("final_convolution")
#
#initial_convolution=ops.create().imgPlus(initial_convolution)
#initial_convolution.setName("initial_convolution")
#
#final_black_white=ops.create().imgPlus(convolved2)
#final_black_white.setName("final_black_white")
# NOTES!!!!

#initial gaussian, create new kernel based on it change values tiny bit.
#2 kernels a bit different image, convert their outputs into a classification
#if new kernel is strictly better, then change
# jython, imageJ, imglib2

image1=ops.create().imgPlus(classification[0])
image1.setName("image1")

image2=ops.create().imgPlus(classification[1])
image2.setName("image2")

image3=ops.create().imgPlus(classification[2])
image3.setName("image3")

#update error to put weights on each image, update weights
#keep track of each image after each boosting phase and add to some image
#have two differnet weights
# w is the weight of each image for calculating error
# a is the scalar multiple to multiply pixels of convolved1 and add to final_image
# exponent of "update weights" section on wiki page is multiplying a by the score