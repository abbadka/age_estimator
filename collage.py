# source: https://stackoverflow.com/questions/35438802/making-a-collage-in-pil
from PIL import Image
import os
location = './data/cropped_faces'
images = [name for name in os.listdir(location)]
images = images[:1200]


def create_collage(width, height, images):
    cols = 20
    rows = 60
    thumbnail_width = width//cols
    thumbnail_height = height//rows
    size = thumbnail_width, thumbnail_height
    new_im = Image.new('RGB', (width, height))
    ims = []
    for p in images:
        im = Image.open(location + '/' + p)
        im.thumbnail(size)
        ims.append(im)
        print(p)
    i = 0
    x = 0
    y = 0
    for col in range(cols):
        for row in range(rows):
            print(i, x, y)
            new_im.paste(ims[i], (x, y))
            i += 1
            y += thumbnail_height
        x += thumbnail_width
        y = 0

    new_im.save("collage.jpg")


create_collage(400, 1200, images)
