import torch
from dataset import AmazonDataset
from torch.utils.data import DataLoader
from unet import UNet
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score


root_dir = "d:/Utente/Desktop/AmazonForest proj/ATLANTIC FOREST"

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
     # sia sul  validation set di "ATLANTIC FOREST", che raccoglie
     # 100 immagini 4x512x512 della foresta atlantica.
    
    dataset_test = AmazonDataset(mode='test')
    dataloader_test = DataLoader(dataset_test, batch_size=4, shuffle=False)

    
    model = UNet(in_channels=4, out_channels=1)

    #TODO: testare il modello su dataset diverso: ATLANTIC FOREST
    model.load_state_dict(torch.load("best_model.pth"))
    print("Modello caricato.")

    
    test_loss, precision, recall, f1, iou = test_unet(model, dataloader_test)
    print(f"Test completato. Loss: {test_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, IoU: {iou:.4f}")