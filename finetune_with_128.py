# -*- coding: utf-8 -*-

"""
전체 학습 과정은 아래와 같습니다.
1. WideResNet을 CIFAR-100으로 pretrain
2. pretrain한 모델을 CUB-200으로 Fine-Tuning
  * a. CUB-200을 저해상도로 낮춘 데이터(128*128)를 학습
  * b. CUB-200(224*224)를 학습

본 파일에는 2.a.에 대한 코드와 최종 성능을 확인하는 코드가 포함되어 있으며, 1과 2.b에 대한 파일은 깃헙에 포함되어 있습니다
"""

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
import torch.nn.functional as F
from tqdm.auto import tqdm

from PIL import Image


"""# 2. Load the Caltech UCSD Birds-200 Dataset"""

BATCH_SIZE = 64

torch.manual_seed(42)
np.random.seed(42)

class CUB_Dataset(Dataset):
    def __init__(self,img_file, label_file, transform=None):
        self.img =np.load(img_file)
        self.labels = np.load(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = self.img[idx]
        label = self.labels[idx]

        if image.max() <= 1.0:
            image = (image * 255).astype('uint8')

        # NumPy 배열을 PIL 이미지로 변환
        image = Image.fromarray(image)


        if self.transform:
            image = self.transform(image)


        return image,label

cub_bird_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

### transform for train data
cub_bird_transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("data loading start!")

cub_train_dataset = CUB_Dataset(img_file="../data/CUB_train_images.npy",
                                        label_file="../data/CUB_train_labels.npy",transform=cub_bird_transform_train) ### transform for training data (128*128)
cub_finetune_dataset = CUB_Dataset(img_file="../data/CUB_train_images.npy",
                                        label_file="../data/CUB_train_labels.npy",transform=cub_bird_transform) ### transform for finetuning data (224*224)

cub_train_loader = torch.utils.data.DataLoader(cub_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

cub_finetune_loader = torch.utils.data.DataLoader(cub_finetune_dataset, batch_size=BATCH_SIZE, shuffle=True)

cub_val_dataset = CUB_Dataset(img_file="../data/CUB_val_images.npy",
                                        label_file="../data/CUB_val_labels.npy",transform=cub_bird_transform)
cub_val_loader = torch.utils.data.DataLoader(cub_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Number of samples in the dataset
print("caltech bird train dataset size : ", len(cub_train_dataset))
print("caltech bird finetune dataset size : ", len(cub_finetune_dataset))
print("caltech bird validation dataset size : ", len(cub_val_dataset))

"""## Caltech UCSD Birds-200 Visualiztion"""

# Plot the training images and labels

cub_denormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
to_pil_image = transforms.functional.to_pil_image

# images, labels = next(iter(cub_train_loader))
images, labels = next(iter(cub_finetune_loader))
# images, labels = next(iter(cub_val_loader))

fig, ax = plt.subplots(1, 4, figsize=(16, 4))
ax[0].imshow(to_pil_image(cub_denormalize(images[0])))
ax[1].imshow(to_pil_image(cub_denormalize(images[1])))
ax[2].imshow(to_pil_image(cub_denormalize(images[2])))
ax[3].imshow(to_pil_image(cub_denormalize(images[3])))
plt.show()

print(labels[:4])

"""# 3. Define the Model Architecture
"""

from collections import OrderedDict
import torch
import importlib

pretrain_config = OrderedDict([
    ('arch', 'wrn'),  # Wide ResNet
    ('depth', 28),
    ('base_channels', 16),
    ('widening_factor', 10),
    ('drop_rate', 0.3),
    ('input_shape', (1, 3, 32, 32)),
    ('n_classes', 100),
])

def load_model(config):
    module = importlib.import_module(config['arch'])
    Network = getattr(module, 'Network')
    return Network(config)

# 모델 초기화 : Wide ResNet
model = load_model(pretrain_config)

# Pretrained weights 로드
checkpoint = torch.load("model_wrn_100_epoch200.pt")
print(f"checkpoint.keys(): {checkpoint.keys()}")
model.load_state_dict(checkpoint['state_dict'])
# 출력 개수 맞추기
model.fc = torch.nn.Linear(model.fc.in_features, 200)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

correct = 0
total = 0

with torch.no_grad():
    for data in cub_val_loader:
        images, labels = data
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum().item()

print(f'Finetune 이전) Accuracy of the network on the 2897 validation images: {100 * correct / total:.2f} %')



"""# 4. Train the network"""

import warnings
warnings.filterwarnings('ignore')
def train(model, epochs, train_loader, criterion, optimizer, device, scheduler=None):
    print("Training started!")

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(tqdm(train_loader)):
            inputs, targets = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate running loss
            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Update learning rate if a scheduler is provided
        if scheduler:
            scheduler.step()

        if (epoch+1)%10==0:
            state = OrderedDict([
                ('state_dict', model.state_dict()),
                ('optimizer', optimizer.state_dict()),
                # ('scheduler', scheduler.state_dict()),
                ('epoch', epoch)
            ])
            torch.save(state, f"./model_train_cub_epoch{epoch+1}.pt") # pretrain한 모델 파일 (.pt) 저장

    print("Training complete!")



# finetuning with low resolution
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
train(model, epochs=30, train_loader=cub_train_loader, criterion=criterion, optimizer=optimizer, device=device)


"""# 5. Evaluate the network on the validation data"""

correct = 0
total = 0
with torch.no_grad():
    for data in cub_val_loader:
        images, labels = data
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum().item()


print(f'low freq 학습 이후) Accuracy of the network on the 2897 validation images: {100 * correct / total:.2f} %')