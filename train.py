import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm
from tqdm import tqdm
import os
import copy
import time
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

import matplotlib.pyplot as plt

# Configuration
DATA_DIR = "dataset_split"
MODEL_NAME = "xception"
NUM_CLASSES = 1 # Binary classification
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10
IMG_SIZE = 299
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_history(history):
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    
    epochs = range(len(acc))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    print("Training history saved to training_history.png")


def get_data_loaders(data_dir, img_size, batch_size):
    # Data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Xception expects -1 to 1 usually or standard scaling
            # timm models usually expect ImageNet stats, let's stick to standard mean/std if possible or simple norm
            # standard: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4)
                   for x in ['train', 'val', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

def build_model(model_name, num_classes):
    print(f"Creating model: {model_name}")
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} epoch {epoch}"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).float().unsqueeze(1) # BCSWithLogitsLoss expects float labels

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    preds = torch.sigmoid(outputs) > 0.5

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            try:
                epoch_auc = roc_auc_score(all_labels, all_preds)
            except:
                epoch_auc = 0.0

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.cpu().numpy())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.cpu().numpy())

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss: # Saving based on Val Loss
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model.pth')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f} Loss: {best_loss:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Dataset directory {DATA_DIR} not found.")
        return

    dataloaders, dataset_sizes, class_names = get_data_loaders(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    print(f"Classes: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")
    
    model = build_model(MODEL_NAME, NUM_CLASSES)
    if model:
        model = model.to(DEVICE)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        trained_model, history = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=EPOCHS)
        
        # Save final model
        torch.save(trained_model.state_dict(), 'final_model.pth')
        
        # Plot history
        plot_history(history)


if __name__ == "__main__":
    main()
