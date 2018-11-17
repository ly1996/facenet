import cv2
import image
import re
from PIL import Image
from PIL import ImageEnhance
import glob as gb
img_path = gb.glob(r"image_org/*.jpg")
for path in img_path:
    im = Image.open(path)
    result = re.findall(r"image_org/(.+).jpg",path)
    im_path = 'image_/' + result[0] + '.jpg'
    im.save(im_path, 'JPEG')
    dst_flip_path = 'image_/'+result[0]+'_flip.jpg'
    dst_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
    dst_flip.save(dst_flip_path,'JPEG')
    for i in range(-4,5):
        if i != 0:
            dst_rotate_path = 'image_/' + result[0] + '_rotate' + str(10*i) + '.jpg'
            dst_rotate = im.rotate(10*i)
            dst_rotate.save(dst_rotate_path, 'JPEG')
            dst_flip_rotate_path = 'image_/' + result[0] + '_flip_rotate' + str(10 * i) + '.jpg'
            dst_flip_rotate = dst_flip.rotate(10 * i)
            dst_flip_rotate.save(dst_flip_rotate_path, 'JPEG')

