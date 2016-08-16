import struct
from matplotlib import pyplot as plt
import numpy as np
import pylab

def showImages(images, labels, amt):
    if amt > 30:
        amt = 30

    fig = plt.figure(figsize=(10, 20))
    for i in range(amt):
        sp = fig.add_subplot(10, 5, i + 1)
        l = [j for j, x in enumerate(labels[i]) if x][0]
        sp.set_title(l)
        plt.axis('off')
        image = np.array(images[i]).reshape(28, 28)
        plt.imshow(image, interpolation='none', cmap=pylab.gray(), label=l)
    plt.show()


def showSingleImage(image):
    image = np.array(image).reshape(28, 28)
    plt.imshow(image, interpolation='none', cmap=pylab.gray())
    plt.show()