import torchvision
import torchvision.transforms as transforms
from skimage import io, color
import sys
import numpy as numpy

filename=sys.argv[1]
img = io.imread(filename)
print(type(img))
img = transforms.functional.to_pil_image(img)
img=transforms.functional.to_grayscale(img)
img=numpy.array(img)
io.imsave(fname=filename,arr=img)
