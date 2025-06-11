import torch
from torch.utils.data import Dataset
import rasterio
import os
from osgeo import gdal
from utils import imgread
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

root_dir = "d:/Utente/Desktop/Amazon forest proj/AMAZON"


class AmazonDataset(Dataset):
    def __init__(self, root_dir=root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        
        self.transform = transform
        self.mode = mode

        if mode == 'train':
            self.root_dir = os.path.join(root_dir, 'Training')
        elif mode == 'val':
            self.root_dir = os.path.join(root_dir, 'Validation')
        elif mode == 'test':
            self.root_dir = os.path.join(root_dir, 'Test')
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'.")
        
        self.image_dir = os.path.join(self.root_dir, 'image')
        self.label_dir = os.path.join(self.root_dir, 'label')
        self.image_files = os.listdir(self.image_dir)
    def __len__(self):
        """
        Restituisce il numero di immagini nel dataset.
        """
        return len(self.image_files)
    

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx])
        #img=imgread(image_path)
        img = rasterio.open(image_path).read().astype(np.float32)
        label = rasterio.open(label_path).read().astype(np.float32)
        img=img/10000
        #img = img.astype(np.float32)
        #mg = gdal.Open(image_path)

        '''
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
        
        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.int64)

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image), torch.tensor(label)
        '''

        return img, label
    

dataset= AmazonDataset(root_dir=root_dir, mode='train')
image , label=dataset[0]
#print(image)  # Stampa la forma dell'immagine per verificare che sia corretta
print(image)
print(np.unique(label))  # Stampa i valori unici dell'etichetta per verificare che siano corretti
def multispectral_to_rgb_visualization(img, lower_percentile=5, upper_percentile=95):


    assert isinstance(img, np.ndarray), "The input image must be a numpy array"
    img = img.transpose(1,2,0)
    img = img[:, :, [2, 1, 0]]
    img = np.clip(img, np.percentile(img, lower_percentile), np.percentile(img, upper_percentile))
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img * 255).astype(np.uint8)
    plt.imshow(img)
    plt.axis('off')  # Rimuove gli assi per una visualizzazione più pulita
    plt.title("Multispectral Image Visualization")
    plt.show()
    return img
#img= multispectral_to_rgb_visualization(image)
#print(img)