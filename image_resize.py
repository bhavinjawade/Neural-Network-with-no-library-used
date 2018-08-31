from PIL import Image
import numpy as np

from resizeimage import resizeimage
import matplotlib.image as img

with open('number.jpg', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [16, 16])
        cover.save('number2.jpg', image.format)

image = img.imread('number2.jpg')
im = image
print(image)
print(len(image))

listfinal = []
for i in range(len(im)):
    for j,list in enumerate(im[i]):
        intensity = 0.2989*list[0] + 0.5870*list[1] + 0.1140*list[2]
        print(str(list[0]) + "," + str(list[1]) + "," + str(list[2]) + "-" + str(intensity))
        if(intensity >= 128):
            listfinal.append(0)
        else:
            listfinal.append(1)

print(np.array(listfinal))