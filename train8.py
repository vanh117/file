from torchvision import transforms
import dataset # Custom dataset
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet101, ResNet101_Weights
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
# def get_random_subset_dataloader(dataset, subset_size, batch_size=32):
#     # Randomly sample indices with replacement
    
#     subset_indices = np.random.choice(len(dataset), size=subset_size, replace=True)
#     subset = torch.utils.data.Subset(dataset, subset_indices)
#     dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)
#     return dataloader
import os 
root_dir = 'final_1'
# make dir if not exist
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

model_nths = [0]
batch_size = 4
# batch_size = 64 
train_log_dir = 'train_log'
train_log_dir = os.path.join(root_dir, train_log_dir)
make_dir(train_log_dir)

val_log_dir = 'val_log'
val_log_dir = os.path.join(root_dir, val_log_dir)
make_dir(val_log_dir)




transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# test 


# Load your dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensemble_predict(models, dataloader):
    all_outputs = []
    all_labels = []
    for model in models:
        model.eval()
    
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = [model(inputs) for model in models]
            avg_outputs = torch.mean(torch.stack(outputs), dim=0)
            all_outputs.append(avg_outputs)
            all_labels.append(labels)
            
# torch.no_grad() -> không cập nhật đạo hàm trong quá trình eval()
# line 64 -> gọi models -> check 
    # Concatenate all batches
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_outputs, all_labels

def calculate_accuracy(predictions, labels):
    _, predicted_classes = predictions.max(dim=1)
    correct = (predicted_classes == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


def test(models, test_dataset, log_file):
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    all_predictions, all_labels = ensemble_predict(models, test_loader)
    test_accuracy = calculate_accuracy(all_predictions, all_labels)
    with open(log_file, 'a') as f:
        f.write(f"Accuracy: {test_accuracy * 100:.2f}% \n")
    return test_accuracy

def validation(model, val_dataset, log_file, epoch):
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    val_loss = 0 
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
    val_loss /= len(val_loader)

    with open(log_file, 'a') as f:
        f.write(f"Epoch: {epoch}, val_loss: {val_loss}\n")
    return val_loss

def train_model(model, dataloader, criterion, optimizer, num_epochs=40, model_fold=0):

    train_loss_log = os.path.join(train_log_dir, f'train_loss_{model_fold}.txt')
    val_loss_log = os.path.join(val_log_dir, f'val_loss_{model_fold}.txt')
    
    val_dataset = dataset.CustomImageDataset(annotations_file= os.path.join(root_dir, f'fold{model_fold}_val.txt') , img_dir='final_train/images', transform=transform)
    
    # tqdm : calculate time for 1 epoch
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # log to txt file 
            total_loss += loss.item()
        total_loss /= len(dataloader)
    
        with open(train_loss_log, 'a') as f:
            f.write(f'Epoch {epoch}, train_loss: {total_loss}\n')
        validation(model, val_dataset, val_loss_log, epoch)
        test([model], val_dataset, val_loss_log)
        # [model] 
        
# Parameters
# model = resnet101(weights=ResNet101_Weights.DEFAULT)
# model.load_state_dict(torch.load('resnet101_model_0.pth'))

for i in model_nths :
    model = resnet101(weights=ResNet101_Weights.DEFAULT) 
    # load pretrain model
    model.fc = nn.Linear(model.fc.in_features, 150)  # Adjust for 150 classes
    # tạo model -> 
    # model.load_state_dict(torch.load(os.path.join(root_dir, f'resnet101_model_{i}.pth')))
    # model.load_state_dict(torch.load('resnet101_model_0.pth'))
    # 142, 143 là đã pretrain --> run tiếp
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()

    # Get a unique DataLoader for each model
    
    train_dataset = dataset.CustomImageDataset(annotations_file=os.path.join(root_dir, f'fold{i}_train.txt'), img_dir='final_train/images', transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Training model {i}") 
    train_model(model, dataloader, criterion, optimizer, num_epochs=30, model_fold=i)
    

    
    
    # Save model to disk
    torch.save(model.state_dict(), root_dir + f'/resnet101_model_{i}.pth')
    # clear cache
    del model
    torch.cuda.empty_cache()