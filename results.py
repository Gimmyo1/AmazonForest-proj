
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import AmazonDataset
from unet import UNet
from torch.utils.data import DataLoader

def multispectral_to_rgb_visualization(img, lower_percentile=5, upper_percentile=95):
    assert isinstance(img, np.ndarray), "The input image must be a numpy array"
    img = img.transpose(1, 2, 0)
    img = img[:, :, [2, 1, 0]]
    img = np.clip(img, np.percentile(img, lower_percentile), np.percentile(img, upper_percentile))
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img * 255).astype(np.uint8)
    return img  # non mostrare qui, lo faremo dopo

def visualize_predictions(model, dataset, num_samples=4, save_path="qualitative_results.png"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    for i in range(num_samples):
        image, label = dataset[i]
        image_tensor = image.unsqueeze(0).to(device)
        label_tensor = label.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = (torch.sigmoid(output) > 0.5).int().squeeze().cpu().numpy()

        label = np.uint8(label)

        
        rgb_image = multispectral_to_rgb_visualization(np.array(image))

        # Calcolo mappa degli errori
        tp = (label == 1) & (pred_mask == 1)
        tn = (label == 0) & (pred_mask == 0)
        fn = (label == 1) & (pred_mask == 0)
        fp = (label == 0) & (pred_mask == 1)

        error_map = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        error_map[tn] = [0, 0, 0]         # Nero
        error_map[tp] = [255, 255, 255]   # Bianco
        error_map[fn] = [255, 0, 0]       # Rosso
        error_map[fp] = [0, 0, 255]       # Blu

        # Plot
        axes[i, 0].imshow(rgb_image)
        axes[i, 0].set_title("Input RGB")

        axes[i, 1].imshow(pred_mask, cmap='gray')
        axes[i, 1].set_title("Predicted Mask")

        axes[i, 2].imshow(label, cmap='gray')
        axes[i, 2].set_title("Ground Truth")

        axes[i, 3].imshow(error_map)
        axes[i, 3].set_title("Error Map")

        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✔️ Risultato salvato in: {save_path}")

dataset_amazon = AmazonDataset(mode='test')
dataloader_amazon = DataLoader(dataset_amazon, batch_size=4, shuffle=False)

model = UNet(in_channels=4, out_channels=1)

model.load_state_dict(torch.load("best_model.pth"))
print("Modello caricato.")

visualize_predictions(model, dataset_amazon, num_samples=4, save_path="qualitative_test_results.png")
