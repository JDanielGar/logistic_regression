from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def ProcessImage(image_dir):
    img = Image.open(str(image_dir)).convert('L').resize((48, 48), Image.ANTIALIAS)
    return np.array(list(img.getdata()))/255.0

# label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def ShowImage(data, title):
    label_map = ['Happy', 'Sad']
    plt.imshow(data.reshape(48, 48), cmap='gray')
    plt.title(label_map[1])
    plt.show()
    return print('(+) Successfully printed image.')