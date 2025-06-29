import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from unet import UNet
from dataset import AmazonDataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def train_unet(model, dataloader_train, dataloader_val, num_epochs=10, learning_rate=0.001):
    
    device = torch.device( "cuda")
    model.to(device)
    
   
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    best_iou = 0
    train_losses = []
    val_losses = []
    iou_scores = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in dataloader_train:

            images = images.to(device)
            labels = labels.unsqueeze(1).to(device) 

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item():.4f}")
        avg_loss = running_loss / len(dataloader_train)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

       
        model.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for images, labels in dataloader_val:
                images = images.to(device)
                labels = labels.unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).int()
                y_true.extend(labels.cpu().numpy().flatten())
                y_pred.extend(preds.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(dataloader_val)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        # metriche
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        iou = jaccard_score(y_true, y_pred)
        iou_scores.append(iou)
        

        print(f"Epoch [{epoch+1}/{num_epochs}], Validation mIoU: {iou:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

       
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), "best_model.pth")
            print("Miglior modello salvato!")


        

    
    
    print("Training complete.")


    # Plot delle loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("loss_plot.png")
    plt.show()


    return model




if __name__ == "__main__":

    

    
    dataset_train = AmazonDataset(mode='train')
    dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)

    dataset_val = AmazonDataset(mode='val')
    dataloader_val = DataLoader(dataset_val, batch_size=4, shuffle=False)

    model = UNet(in_channels=4, out_channels=1)
    
    trained_model = train_unet(model, dataloader_train, dataloader_val, num_epochs=10, learning_rate=0.0001)
    
    # Salva il modello addestrato
    torch.save(trained_model.state_dict(), "unet_trained.pth")