# @OpService ops
# @DatasetIOService ds
# @OUTPUT ImgPlus image1
# @OUTPUT ImgPlus image2
# @OUTPUT ImgPlus image3

from net.imglib2 import Point

from net.imglib2.algorithm.region.hypersphere import HyperSphere

from ij import IJ

from mpicbg.imglib.interpolation.nearestneighbor import NearestNeighborInterpolatorFactory;

import random

preys = []
classification = []
# kernels = []
for i in range(1,4):
	if i == 0:
		preys.append(ops.create().imgPlus(ds.open("/Users/test/Desktop/Git/Predator_vs_Prey/prey_images/grass_greyscale.jpg")))
		curr_image = ops.create().imgPlus(ds.open("/Users/test/Desktop/Git/Predator_vs_Prey/prey_images/grass_greyscale.jpg"))
	
		while (cursor_curr.hasNext()):
			cursor_curr.fwd()
			cursor_curr.get().set(0)
		classification.append(curr_image)
	else:
		preys.append(ops.create().imgPlus(ds.open("/Users/test/Desktop/Git/Predator_vs_Prey/prey_images/generation0000%d_max_prey.png" % (i * 3))))
		classification.append(ops.create().imgPlus(ds.open("/Users/test/Desktop/Git/Predator_vs_Prey/prey_images/classification.png")))

xSize = 250
ySize = 250

for a in range(5):
	kernel_init=ops.create().kernelGauss([16, 16])
	kernel_mod=ops.copy().img(kernel_init)
				
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
		if (score2 > score1):
				kernel_init = ops.copy().img(kernel_mod)
		else:
				kernel_mod = ops.copy().img(kernel_init)
					
	# kernels.append(kernel_init)
	
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

image1=ops.create().imgPlus(classification[0])
image1.setName("image1")

image2=ops.create().imgPlus(classification[1])
image2.setName("image2")

image3=ops.create().imgPlus(classification[2])
image3.setName("image3")