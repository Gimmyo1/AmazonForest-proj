import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from unet import UNet
from dataset import AmazonDataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def train_unet(model, dataloader_train, dataloader_val, num_epochs=10, learning_rate=0.001):
    
    device = torch.device( "cuda")
    model.to(device)
    
   
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_miou = 0
    train_losses = []
    val_losses = []
    miou_scores = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in dataloader_train:

            images = images.to(device)
            labels = labels.to(device) 

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item():.4f}")
        avg_loss = running_loss / len(dataloader_train)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

       
        model.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in dataloader_val:
                images = images.to(device)
                labels = labels.to(device) # Assicurati che le etichette siano della forma corretta

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).int()
                y_true.extend(labels.cpu().numpy().flatten())
                y_pred.extend(preds.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(dataloader_val)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        # Calcolo delle metriche
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        miou = jaccard_score(y_true, y_pred)
        miou_scores.append(miou)

        #num_classes = 2
        #iou_per_class, miou = calculate_iou_per_class(np.array(y_true), np.array(y_pred), num_classes)



        print(f"Epoch [{epoch+1}/{num_epochs}], Validation mIoU: {miou:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

       
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), "best_model.pth")
            print("Miglior modello salvato!")


        

    
    
    print("Training complete.")
    return model




if __name__ == "__main__":
    # Esempio di utilizzo
    

    
    dataset_train = AmazonDataset(mode='train')
    dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=True)

    dataset_val = AmazonDataset(mode='val')
    dataloader_val = DataLoader(dataset_val, batch_size=2, shuffle=False)

    model = UNet(in_channels=4, out_channels=1)
    
    trained_model = train_unet(model, dataloader_train, dataloader_val, num_epochs=10, learning_rate=0.0001)
    
    # Salva il modello addestrato
    torch.save(trained_model.state_dict(), "unet_mados.pth")