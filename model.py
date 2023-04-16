import pydicom
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the CNN model with attention maps
class CNNWithAttention(nn.Module):
    def __init__(self):
        super(CNNWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.attention = nn.Conv2d(64, 1, kernel_size=1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        att = torch.sigmoid(self.attention(x))
        x = torch.mul(att, x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the data preprocessing and augmentation
transform = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(15),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# Load the DICOM images
dataset = datasets.ImageFolder('path/to/dicom/images', transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Initialize the CNN model and optimizer
model = CNNWithAttention()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Train the model
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_dataset)
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_dataset)
    
    print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'
          .format(epoch+1, num_epochs, train_loss, val_loss))

# Save the trained model
torch.save(model.state_dict(), "trained_model.pth")

#show example of an image and the attention map


dicom_image = pydicom.read_file('/home/dima/UOC/data/manifest-1608669183333/Lung-PET-CT-Dx/Lung_Dx-A0263/02-12-2010-NA-PET07PTheadlung Adult-81028/10.000000-Thorax  1.0  B70f-60855/1-001.dcm')

# Extract pixel data from DICOM image
pixel_array = dicom_image.pixel_array.astype(np.float)

# Convert pixel data to torch tensor
image_tensor = torch.tensor(pixel_array).unsqueeze(0).unsqueeze(0).float()
# Forward pass through the CNN to obtain attention maps
output, attention_map = model(image_tensor)

# Convert attention map to numpy array for visualization
attention_map = attention_map.detach().numpy()[0, 0, :, :]

# Display the original image and attention map
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(pixel_array, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(attention_map, cmap='gray')
axs[1].set_title('Attention Map')
axs[1].axis('off')
plt.show()