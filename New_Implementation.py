# @OpService ops
# @DatasetIOService ds
# @UIService ui
# @OUTPUT ImgPlus image1
# @OUTPUT ImgPlus image2
# @OUTPUT ImgPlus image3

# TODO: CHECK OLD IMPLEMENTATION WORKS WITH DIFFERENTLY ORIENTED PREY
# TODO: PEROFORM TESTING ON OLD IMPLEMTNATION (STACK KERNELS)
# TODO: INCREASE BOOSTING TIMES AND KERNEL SEARCHING TIMES ON NEW IMPLEMENTATION

from net.imglib2 import Point

from net.imglib2.algorithm.region.hypersphere import HyperSphere

from ij import IJ

from mpicbg.imglib.interpolation.nearestneighbor import NearestNeighborInterpolatorFactory;

import random
import math

def xor(a,b):
	return (a > THRESHOLD and b > THRESHOLD) or (a <= THRESHOLD and b <= THRESHOLD)

X_DIM = 250
Y_DIM = 250
NUM_IMAGES = 3
NUM_BOOSTING = 3
NUM_KERNEL_SEARCHING = 100
THRESHOLD = 0
#95

preys = []
classification = []
#TODO: CHANGE THIS TO CURSORS
#TODO: POTENTIALLY MULTIPLE BY NUM_IMAGES TO MAKE BEST ERROR CALC NOT AN AVERAGE
error_weights = [[1.0/(X_DIM * Y_DIM) for i in range(X_DIM * Y_DIM)] for j in range(NUM_IMAGES)]
final_images = []

#make initial "ensemble" a blank image		
for i in range(NUM_IMAGES):
	curr_image = ds.open("/Users/test/Desktop/Git/Predator_vs_Prey/prey_images/grass_greyscale.jpg").getImgPlus()
	cursor_curr=curr_image.cursor()

	while ( cursor_curr.hasNext()):
		cursor_curr.fwd()
		cursor_curr.get().set(0)
	curr_image = ops.convert().float32(curr_image)
	final_images.append(curr_image)
	

#load relevant images
for i in range(1,NUM_IMAGES + 1):
#	if (i == 1):
#		curr_image = ds.open("/Users/test/Desktop/Git/Predator_vs_Prey/prey_images/grass_greyscale.jpg").getImgPlus()
#		cursor_curr=curr_image.cursor()
#		
#		while ( cursor_curr.hasNext()):
#			cursor_curr.fwd()
#			cursor_curr.get().set(-1)
#		classification.append(curr_image)
#		curr_image = ds.open("/Users/test/Desktop/Git/Predator_vs_Prey/prey_images/grass_greyscale.jpg").getImgPlus()
#		
#	else:
	curr_image = ds.open("/Users/test/Desktop/Git/Predator_vs_Prey/prey_images/classification.png").getImgPlus()
	curr_image = ops.convert().float32(curr_image)
	cursor_curr=curr_image.cursor()

	while ( cursor_curr.hasNext()):
		cursor_curr.fwd()
		if (cursor_curr.get().get() == 0):
			cursor_curr.get().set(-1.0)
		else:
			cursor_curr.get().set(1.0)
	classification.append(curr_image)
	curr_image = ds.open("/Users/test/Desktop/Git/Predator_vs_Prey/prey_images/generation0000%d_max_prey.png" % (i * NUM_IMAGES)).getImgPlus()
	
	#normalize image
	curr_image = ops.convert().float32(curr_image)
	cursor_image=curr_image.cursor()
	mean = ops.stats().mean(curr_image)
	std_dev = ops.stats().stdDev(curr_image)
	
	while ( cursor_image.hasNext()):
		cursor_image.fwd()
		cursor_image.get().set( (cursor_image.get().get() - mean.get())/ std_dev.get() )
		
	preys.append(curr_image)

#main loop
for i in range(NUM_BOOSTING):
#		TODO: MAKE KERNEL NORMALIZED
	kernel_best = ops.create().kernelGauss([16, 16])
	cursor_image=kernel_best.cursor()
	
	while ( cursor_image.hasNext()):
		cursor_image.fwd()
		cursor_image.get().set(random.uniform(-.1, 0.1))
	
	best_error = NUM_IMAGES + 1.0
	print("First error: %s" % str(error_weights[:2][:10]))
	print("Sum error: %f" % sum(error_weights[0]))
	kernel_mod = ops.copy().img(kernel_best)	

	#look for kernel that maximizes performance
	for j in range(NUM_KERNEL_SEARCHING): 
		cursor_kernel_mod = kernel_mod.cursor()

		#create a potentially better kernel
		while ( cursor_kernel_mod.hasNext()):
				cursor_kernel_mod.fwd()
				cursor_kernel_mod.get().set( cursor_kernel_mod.get().get() + random.uniform(-.001, 0.001))
		
		mod_error = 0.0

		#calculate cumulative error on all images
		for curr_prey_index in range(NUM_IMAGES):	
			convolved_mod=ops.filter().convolve(preys[curr_prey_index], kernel_mod)
			
			cursor_class=classification[curr_prey_index].cursor()
			cursor_convolved_mod=convolved_mod.cursor()

			curr_pixel_index = 0
			while ( cursor_convolved_mod.hasNext()):
				cursor_convolved_mod.fwd()
				cursor_class.fwd()
				if (cursor_convolved_mod.get().get() > THRESHOLD):
						cursor_convolved_mod.get().set( 1.0 )
				else:
						cursor_convolved_mod.get().set( -1.0 )	
				if xor(cursor_convolved_mod.get().get(),cursor_class.get().get()) == False:
						mod_error += 1.0 * error_weights[curr_prey_index][curr_pixel_index]
				curr_pixel_index += 1

		#replace best kernel if outperformed
#		print("Mod error: %f" % mod_error)
#		print("Best error: %f" % best_error)
		if (mod_error < best_error):
				kernel_best = ops.copy().img(kernel_mod)
				best_error = mod_error
		else:
				kernel_mod = ops.copy().img(kernel_best)
					
	#weight to put on convolved image before adding to ensemble	
#	if (best_error == 0):
#		kernel_weight = 1
#	else:
	best_error = best_error / NUM_IMAGES
	kernel_weight = .5 * math.log((1.0 - best_error)/best_error)
	
	for j in range(NUM_IMAGES):
		convolved_best=ops.filter().convolve(preys[j], kernel_best)
		output = ops.filter().convolve(preys[j], kernel_best)
		
		cursor_convolved_best=convolved_best.cursor()
		cursor_final=final_images[j].cursor()
		cursor_class=classification[j].cursor()
		cursor_output = output.cursor()
		
		k = 0
		#add to ensemble, update pixel error weighting
		while ( cursor_final.hasNext()):
				cursor_final.fwd()
				cursor_convolved_best.fwd()
				cursor_class.fwd()
				cursor_output.fwd()

				#used to have type casting to integer
				cursor_final.get().set(cursor_final.get().get() + cursor_convolved_best.get().get() * kernel_weight)
				error_weights[j][k] = error_weights[j][k] * (math.exp(-1.0 * cursor_class.get().get() * cursor_convolved_best.get().get() * kernel_weight))
				if (cursor_final.get().get() > THRESHOLD):
						cursor_output.get().set(1.0)
				else:
						cursor_output.get().set(-1.0)
				k += 1
		ui.show(output)

image1=ops.create().imgPlus(final_images[0])
image1.setName("image1")
image2=ops.create().imgPlus(final_images[1])
image2.setName("image2")
image3=ops.create().imgPlus(final_images[2])
image3.setName("image3")

#TODO: 
# Running old implementation with normalization
# 