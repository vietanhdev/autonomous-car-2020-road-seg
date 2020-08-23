
import random
import numpy as np
import cv2
import random

seq = [None]

def load_aug():

	import imgaug as ia
	from imgaug import augmenters as iaa

	sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	seq[0] = iaa.Sequential(
		[
			# apply the following augmenters to most images
			iaa.Fliplr(0.5), # horizontally flip 50% of all images
			# iaa.Flipud(0.1), # vertically flip 10% of all images
			# crop images by -5% to 10% of their height/width
			sometimes(iaa.CropAndPad(
				percent=(-0.05, 0.05),
				pad_cval=(0, 255)
			)),
			# execute 0 to 5 of the following (less important) augmenters per image
			# don't execute all of them, as that would often be way too strong
			iaa.SomeOf((0, 3),
				[
					iaa.OneOf([
						iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 1.0
						iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 3 and 5
						iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 3 and 5
					]),
					iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5)), # sharpen images
					# search either for all edges or for directed edges,
					iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images
					iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
					iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
					# either change the brightness of the whole image (sometimes
					# per channel) or change the brightness of subareas
					iaa.contrast.LinearContrast((0.8, 1.2), per_channel=0.2), # improve or worsen the contrast
					# iaa.Grayscale(alpha=(0.0, 0.5)),
					sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
				],
				random_order=True
			)
		],
		random_order=True
	)


def _augment_seg( img , seg  ):

	import imgaug as ia

	if seq[0] is None:
		load_aug()

	# Add snow
	if random.random() < 0.2:
		img_h, img_w = img.shape[:2]
		for _ in range(random.randint(200, 1000)):

			overlay = img.copy()
			alpha = random.random()  # Transparency factor.

			y_min = random.randint(img_h * .5, img_h * .8)
			x, y, a, b = random.randint(1, img_w), random.randint(y_min, img_h), random.randint(1, 5), random.randint(1, 5)  # Rectangle parameters
			overlay = cv2.ellipse(overlay, (x, y), (a, b), 45, 0, 360, (255, 255, 255), -1)

			y_min = random.randint(img_h * .5, img_h * .8)
			y_max = random.randint(img_h * .9, img_h)
			x, y, a, b = random.randint(1, img_w), random.randint(y_min, y_max), random.randint(1, 5), random.randint(1, 5)  # Rectangle parameters
			overlay = cv2.ellipse(overlay, (x, y), (a, b), 45, 0, 360, (255, 255, 255), -1)

			# Following line overlays transparent rectangle over the image
			img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

	
	aug_det = seq[0].to_deterministic() 
	image_aug = aug_det.augment_image( img )

	# cv2.imshow("image_aug", image_aug)

	segmap = ia.SegmentationMapsOnImage( seg, shape=img.shape )
	segmap_aug = aug_det.augment_segmentation_maps( segmap )
	segmap_aug = segmap_aug.get_arr()

	# segmap_vis = segmap_aug.copy()
	# segmap_vis[segmap_aug==1] = 255
	# segmap_vis[segmap_aug==2] = 125
	# segmap_vis[segmap_aug==3] = 180
	# cv2.imshow("segmap_vis", segmap_vis)
	# cv2.waitKey(0)

	return image_aug , segmap_aug


def try_n_times( fn , n , *args , **kargs):
	
	attempts = 0

	while attempts < n:
		try:
			return fn( *args , **kargs )
		except Exception as e:
			attempts += 1

	return fn( *args , **kargs )



def augment_seg( img , seg  ):
	return try_n_times( _augment_seg , 10 ,  img , seg  )

