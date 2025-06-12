import torch
from dataset import AmazonDataset
from torch.utils.data import DataLoader
from unet import UNet
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import matplotlib.pyplot as plt
import numpy as np

atlantic_dir = "d:/Utente/Desktop/AmazonForest proj/ATLANTIC FOREST"

def test_unet(model, dataloader_test):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  

    criterion = torch.nn.BCEWithLogitsLoss()
    
    test_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad(): 
        for images, labels in dataloader_test:
            images = images.to(device)
            labels = labels.unsqueeze(1).to(device) 

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # predictions
            preds = (torch.sigmoid(outputs) > 0.5).int()
            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())

    avg_test_loss = test_loss / len(dataloader_test)
    print(f"Test Loss: {avg_test_loss:.4f}")

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    # Precision, Recall, F1-score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Iou
    iou = jaccard_score(y_true, y_pred)

    print(f"Test Accuracy: {(y_true == y_pred).sum().item() / len(y_true):.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"IoU: {iou:.4f}")

    return avg_test_loss, precision, recall, f1, iou


if __name__ == "__main__":

     # il test è effettuato sia sul test set di AMAZON (dataset di training)
     # sia sul  test set di "ATLANTIC FOREST", che raccoglie
     # 100 immagini 4x512x512 della foresta atlantica.
    
    dataset_amazon = AmazonDataset(mode='test')
    dataloader_amazon = DataLoader(dataset_amazon, batch_size=4, shuffle=False)

    
    model = UNet(in_channels=4, out_channels=1)

    model.load_state_dict(torch.load("best_model.pth"))
    print("Modello caricato.")

    
    
    test_loss_a, precision_a, recall_a, f1_a, iou_a = test_unet(model, dataloader_amazon)
    print(f"Test completato. Loss: {test_loss_a:.4f}, Precision: {precision_a:.4f}, Recall: {recall_a:.4f}, F1 Score: {f1_a:.4f}, IoU: {iou_a:.4f}")

    dataset_atlantic = AmazonDataset(root_dir=atlantic_dir, mode='val')  # o 'test' se hai una cartella separata
    dataloader_atlantic = DataLoader(dataset_atlantic, batch_size=4, shuffle=False)

    test_loss_b, precision_b, recall_b, f1_b, iou_b = test_unet(model, dataloader_atlantic)
    print(f"Test completato. Loss: {test_loss_b:.4f}, Precision: {precision_b:.4f}, Recall: {recall_b:.4f}, F1 Score: {f1_b:.4f}, IoU: {iou_b:.4f}")


    metrics = ['Precision', 'Recall', 'F1-score', 'IoU']
    amazon_results = [precision_a, recall_a, f1_a, iou_a]
    atlantic_results = [precision_b, recall_b, f1_b, iou_b]

    # Setup del grafico
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, amazon_results, width, label='Amazon')
    bars2 = ax.bar(x + width/2, atlantic_results, width, label='Atlantic Forest')

    # Etichette
    ax.set_ylabel('Score')
    ax.set_title('Confronto Metriche su Test Set Amazon vs Atlantic Forest')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y')

    # Etichette numeriche sopra le barre
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Salva immagine
    plt.tight_layout()
    plt.savefig('comparison_test_metrics.png')
    plt.show()



