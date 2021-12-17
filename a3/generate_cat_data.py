from PIL import Image
import numpy as np


# with the cats database in the cats/ folder, this script will convert the dataset
# into a numpy object and will save it to the root folder
list = []
for x in range(1, 15747):
    an_image = Image.open('cats/' + str(x) + '.jpg')
    image_sequence = an_image.getdata()
    image_array = np.array(image_sequence)
    list.append(image_array)

arr = np.array(list)
with open('cats.npy', 'wb') as f:
    np.save(f, arr)