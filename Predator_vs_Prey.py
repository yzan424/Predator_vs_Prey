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
import math

def xor(a,b):
	return (a > 0 and b > 0) or (a <= 0 and b <= 0)

X_DIM = 250
Y_DIM = 250
NUM_IMAGES = 3
NUM_BOOSTING = 5
NUM_KERNEL_SEARCHING = 100
THRESHOLD = 95

preys = []
classification = ops.create().imgPlus(ds.open("/Users/test/Desktop/prey_images/classification.png"))
error_weights = [[1/(X_DIM * Y_DIM) for i in range(X_DIM * Y_DIM)] for j in range(NUM_IMAGES)]
final_images = []

#make initial "ensemble" a blank image		
for i in range(NUM_IMAGES):
	curr_image = ops.create().imgPlus(ds.open("/Users/test/Desktop/prey_images/grass_greyscale.jpg"))
	cursor_curr=curr_image.cursor()

	
	while ( cursor_curr.hasNext()):
		cursor_curr.fwd()
		cursor_curr.get().set(0)
	final_images.append(curr_image)

#load relevant images
for i in range(1,NUM_IMAGES + 1):
	if (i == 1):
		curr_image = ops.create().imgPlus(ds.open("/Users/test/Desktop/prey_images/grass_greyscale.jpg"))
	else:
		curr_image = ops.create().imgPlus(ds.open("/Users/test/Desktop/prey_images/generation0000%d_max_prey.png" % (i * NUM_IMAGES)))
	
	#normalize image -- math errors?
#	curr_image = ops.convert()
#	cursor_image=curr_image.cursor()
#	mean = ops.stats().mean(cursor_image)
#	std_dev = ops.stats().stddev(cursor_image)
#	while ( cursor_image.hasNext()):
#		cursor_image.fwd()
#		cursor_image.get().set( (cursor_image.get().get() - mean)/std_dev )
		
	preys.append(curr_image)
#	classification.append(ops.create().imgPlus(ds.open("/Users/test/Desktop/prey_images/classification.png")))

#main loop
for a in range(NUM_BOOSTING):
	kernel_best = ops.create().kernelGauss([16, 16])
	kernel_mod = ops.copy().img(kernel_best)	

	#look for kernel that maximizes performance
	for j in range(NUM_KERNEL_SEARCHING): 
			
		cursor_kernel_best = kernel_best.cursor()
		cursor_kernel_mod = kernel_mod.cursor()

		#create a potentially better kernel
		while ( cursor_kernel_mod.hasNext()):
				cursor_kernel_mod.fwd()
				cursor_kernel_mod.get().set( cursor_kernel_mod.get().get() + random.uniform(-.001, 0.001))
		
		best_error = 0
		mod_error = 0

		#calculate cumulative error on all images
		for curr_prey_index in range(NUM_IMAGES):
			convolved_best=ops.filter().convolve(preys[curr_prey_index], kernel_best)	
			convolved_mod=ops.filter().convolve(preys[curr_prey_index], kernel_mod)
			
			cursor_class=classification.cursor()
			cursor_convolved_best=convolved_best.cursor()
			cursor_convolved_mod=convolved_mod.cursor()

			curr_pixel_index = 0
			while ( cursor_convolved_best.hasNext()):
				cursor_convolved_best.fwd()
				cursor_convolved_mod.fwd()
				cursor_class.fwd()
				if (cursor_convolved_best.get().get() > THRESHOLD):
						cursor_convolved_best.get().set( 1 )
				else:
						cursor_convolved_best.get().set( 0 )
						
				if (cursor_convolved_mod.get().get() > THRESHOLD):
						cursor_convolved_mod.get().set( 1 )
				else:
						cursor_convolved_mod.get().set( 0 )	

				if xor(cursor_convolved_best.get().get(),cursor_class.get().get()) == False:
						best_error += 1 * error_weights[curr_prey_index][curr_pixel_index]
						
				if xor(cursor_convolved_mod.get().get(),cursor_class.get().get()) == False:
						mod_error += 1 * error_weights[curr_prey_index][curr_pixel_index]
				curr_pixel_index += 1

		#replace best kernel if outperformed
		if (mod_error < best_error):
				kernel_best = ops.copy().img(kernel_mod)
		else:
				kernel_mod = ops.copy().img(kernel_best)
					
	cum_error=0
	for j in range(NUM_IMAGES):
		convolved_best=ops.filter().convolve(preys[j], kernel_best)
		
		cursor_convolved_best=convolved_best.cursor()
		cursor_class=classification.cursor()
		
		k = 0

		#get pixel values for new classification image, as well as calculate error on the best convolved image
		while ( cursor_convolved_best.hasNext()):
				cursor_convolved_best.fwd()
				cursor_class.fwd()
				if (cursor_convolved_best.get().get() > THRESHOLD):
						cursor_convolved_best.get().set(1)
				else:
						cursor_convolved_best.get().set(0)
							
				if xor(cursor_convolved_best.get().get(),cursor_class.get().get()) == False:
#						cursor_convolved_best.get().set(0)
						cum_error += 1.0 * error_weights[j][k]
#						cursor_convolved_best.get().set(1)
				k += 1
#		classification[j] = convolved_best
		
	#weight to put on convolved image before adding to ensemble	
	if (cum_error == 0):
		kernel_weight = 1
	else:
		kernel_weight = .5 * math.log((1 - cum_error)/cum_error)
	
	for j in range(NUM_IMAGES):
		convolved_best=ops.filter().convolve(preys[j], kernel_best)
		cursor_convolved_best=convolved_best.cursor()
		cursor_final=final_images[j].cursor()
		k = 0

		#add to ensemble, update pixel error weighting
		while ( cursor_final.hasNext()):
				cursor_final.fwd()
				cursor_convolved_best.fwd()
				print(cursor_final.get().get())
				print(cursor_convolved_best.get().get())
				print(kernel_weight)
				print(cursor_convolved_best.get().get() * kernel_weight)
				print(cursor_final.get().get() + cursor_convolved_best.get().get() * kernel_weight)
				cursor_final.get().set(cursor_final.get().get() + cursor_convolved_best.get().get() * kernel_weight)
				error_weights[j][k] = error_weights[j][k] * (math.exp(-1 * classifications[j][k] * cursor_convolved_best.get().get() * kernel_weight))
				k += 1

image1=ops.create().imgPlus(final_images[0])
image1.setName("image1")
image2=ops.create().imgPlus(final_images[1])
image2.setName("image2")
image3=ops.create().imgPlus(final_images[2])
image3.setName("image3")
#update error to put weights on each image, update weights
#keep track of each image after each boosting phase and add to some image
#have two differnet weights
# w is the weight of each image for calculating error