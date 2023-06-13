import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as datasets
import os
import glob
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

# set up class to get images
class RGBDepthDataset(data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        rgb_path, depth_path, label = self.image_paths[index]
        rgb_image = Image.open(rgb_path).convert('RGB')
        depth_image = Image.open(depth_path).convert('L')  # Convert depth image to grayscale

        if self.transform is not None:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)
        
        return rgb_image, depth_image, torch.tensor(label - 1, dtype=torch.long)  # Convert label to a scalar tensor. Model expects index starting at 0 so -1 included


    def __len__(self):
        return len(self.image_paths)
    
class PillClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super(PillClassifier, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 10 * 10, 128),
            nn.ReLU()
        )
        
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x




num_classes = 15

## UNCOMMENT TO USE RESNET18
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#model.fc = nn.Linear(model.fc.in_features, num_classes)

## UNCOMMENT TO USE CUSTOM MODEL
model = PillClassifier()

# split data into test and train
data_root = './pills'
test_ratio = 0.2 # ratio for splitting into test and train
# Generate folder numbers as labels
pill_folders = [f"pill_{i}" for i in range(1, num_classes + 1)]
image_paths = [] # initialize with empty list
for pill_folder in pill_folders:
    # get files from each folder
    folder_path = os.path.join(data_root, pill_folder)
    # Use glob to find the image files matching the pattern
    rgb_files = sorted(glob.glob(os.path.join(folder_path, 'rgb_*.jpg')))
    depth_files = sorted(glob.glob(os.path.join(folder_path, 'depth_*.jpg')))
    
    assert len(rgb_files) == len(depth_files), f"Number of RGB and depth images do not match in {pill_folder} folder."

    # Collect the paths of RGB and depth images
    for i in range(len(rgb_files)):
        rgb_path = rgb_files[i]
        depth_path = depth_files[i]
        image_paths.append((rgb_path, depth_path))

# Shuffle the image paths within each pill folder
random.shuffle(image_paths)

# Create empty lists to store train and test image paths
train_image_paths = []
test_image_paths = []

# Split the image paths into train and test sets for each pill folder
for pill_folder in pill_folders:
    # Find the image paths corresponding to the current pill folder and assign labels
    folder_image_paths = [(rgb_path, depth_path, int(pill_folder.split('_')[1])) for (rgb_path, depth_path) in image_paths if pill_folder in rgb_path]

    # Split the current pill folder's image paths into train and test sets
    folder_train_image_paths, folder_test_image_paths = train_test_split(folder_image_paths, test_size=test_ratio, random_state=42, shuffle=True, stratify=[label for _, _, label in folder_image_paths])
    
    # Append the train and test image paths of the current pill folder to the overall lists
    train_image_paths.extend(folder_train_image_paths)
    test_image_paths.extend(folder_test_image_paths)

# Print the number of samples in train and test sets
print(f"Number of train samples: {len(train_image_paths)}")
print(f"Number of test samples: {len(test_image_paths)}")

# Set up train and test data
train_data = RGBDepthDataset(train_image_paths, transform=transforms.Compose([
    transforms.ToTensor(),
]))

test_data = RGBDepthDataset(test_image_paths, transform=transforms.Compose([
    transforms.ToTensor(),
]))

# Create Data Loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, drop_last=True)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train Model
num_epochs = 1 # update later to more

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    train_loss = 0.0
    
    model.train()
    for rgb_images, depth_images, labels in train_loader:
        rgb_images = rgb_images.to(device)
        depth_images = depth_images.to(device)
        labels = labels.to(device)
    
        optimizer.zero_grad()
    
        outputs = model(rgb_images)
        loss = criterion(outputs, labels)
    
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item() * rgb_images.size(0)


    train_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")

# Evaluate Model
model.eval()

correct = 0
total = 0
predictions = []
true_labels = []

with torch.no_grad():
    for rgb_images, depth_images, labels in test_loader:
        rgb_images = rgb_images.to(device)
        depth_images = depth_images.to(device)
        labels = labels.to(device)

        outputs = model(rgb_images)
        _, predicted = torch.max(outputs.data, 1)
    
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())


accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")

# Create a confusion matrix
confusion_mat = confusion_matrix(true_labels, predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(confusion_mat, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Display the class labels on the x and y axis
num_classes = len(pill_folders)
# Define the class labels based on the number of classes
class_labels = [f"pill_{i+1}" for i in range(num_classes)]
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)

# Fill the confusion matrix cells with the counts
thresh = confusion_mat.max() / 2
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, f'{confusion_mat[i, j]}', ha='center', va='center', color='white' if confusion_mat[i, j] > thresh else 'black')

plt.tight_layout()
plt.show()