import matplotlib.pyplot as plt
import random
import numpy as np
def disp_generated(figure):
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='GnBu_r')
    plt.show()



# with open('figures.h','rb') as o:  figures = np.load(o)
with open('/Users/rodneymelchers/Dropbox/Private/RGM ORION/Imperium Accounting/tutorials/VAEs/figimported.h','rb') as o:  figures = np.load(o)
disp_generated(figures)