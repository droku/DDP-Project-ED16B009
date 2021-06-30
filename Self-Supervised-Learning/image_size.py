import numpy as np 
from PIL import Image
import cv2
import shutil
import os

source_folder = "data/valid/"
destin_folder = "images/val/"
images_list = os.listdir(source_folder)
print("no of images: ",len(images_list))
i = 0
for image_file in images_list:
	if i%100 == 0:
		print(i%100)
	image = Image.open(source_folder + image_file)
	im_re = image.resize((50,50), Image.ANTIALIAS)
	im_re.save(destin_folder + image_file)
	i = i+1
