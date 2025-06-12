import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from dataset import AmazonDataset

dataset= AmazonDataset(mode='train') 

total_pixels = 0
class_0 = 0
class_1 = 0

for image, label in dataset:
    total_pixels += label.size
    class_0 += np.sum(label == 0)
    class_1 += np.sum(label == 1)


perc_0 = class_0 / total_pixels * 100
perc_1 = class_1 / total_pixels * 100

print(f"Classe 0 (non desertificato): {class_0} pixel ({perc_0:.2f}%)")
print(f"Classe 1 (desertificato): {class_1} pixel ({perc_1:.2f}%)")


plt.figure(figsize=(7,5))
plt.bar(["Classe 0", "Classe 1"], [class_0, class_1], color=["green", "red"])
plt.xlabel("Classi")
plt.ylabel("Numero di pixel")
plt.title("Distribuzione Pixel per Classe")
plt.show()