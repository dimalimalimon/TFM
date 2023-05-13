import numpy as np
import os
from get_data_from_XML import *
from getUID import *
import scipy.ndimage as ndimage
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
from matplotlib import pyplot as plt
import torch.nn as nn
from model import SimpleCancerCNN
np.random.seed(69)


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"
print("Model is running on:",device)

#Transform rgb ctscan to grayscale using:(0.3 * R) + (0.59 * G) + (0.11 * B) 
def ct_to_gray(ctscan):
    r = ctscan[:,:,0]
    g = ctscan[:,:,1]
    b = ctscan[:,:,2]
    graybmp = np.multiply(0.3*r, 0.59*g)
    graybmp = np.multiply(graybmp, 0.11*b)
    return graybmp


class DICOMDataset(Dataset):
    def __init__(self, matrix_list, y_list):
        self.matrix_list = matrix_list
        self.y_list = y_list
        
    def __len__(self):
        return len(self.matrix_list)
    
    def __getitem__(self, idx):
        matrix = self.matrix_list[idx]
        y = self.y_list[idx]
        
        return matrix, y


dicom_path = "/home/dima/UOC/TFM/data/manifest-1608669183333/Lung-PET-CT-Dx/"
annotation_path = "/home/dima/UOC/TFM/Lung-PET-CT-Dx-Annotations-XML-Files-rev12222020/Annotation/"
annon_files = os.listdir(annotation_path)
ctlung_files = os.listdir(dicom_path)
#variables to store data
num_classes = 4
x_all = []
y_all = []

#iterate over ct scans
for subject_name in annon_files:
    if "B" in subject_name: #use subset B only since it is small and we are running stuff locally
        annon_path = annotation_path + subject_name
        ctlung_path = dicom_path + "Lung_Dx-" + subject_name
        
        if not os.path.isdir(ctlung_path):
            print("missing file: ", ctlung_path)
            continue
        lungs = getUID_path(ctlung_path) #get dict with xml:ctscan_num 
        annotations = XML_preprocessor(annon_path, num_classes=num_classes).data
        for k, v in annotations.items():
        
            key = k[:-4] #quitamos xml del nombre
            if key not in lungs:
                print("missing annotation file: ", k)
                continue
            image_data = v[0]
            image_data=[int(i) for i in image_data]
            
            bounding_box = [image_data[0], image_data[1], image_data[2], image_data[3]]
            
            dcm_path, dcm_name = lungs[k[:-4]]
            dicom_image = pydicom.read_file(dcm_path)
            pixel_array = dicom_image.pixel_array.astype(np.float)
            
            if len(pixel_array.shape) == 3: #if it is rgb
                pixel_array = ct_to_gray(pixel_array)
                
            pixel_array = pixel_array / np.max(pixel_array)

            pixel_array = np.expand_dims(pixel_array, axis=-1)

            x_all.append(torch.tensor(pixel_array).to(device))

        #get xmin, ymin, xmax, ymax that define the square and will be predicted
            y_all.append(torch.tensor(np.array([image_data[0], image_data[1], image_data[2], image_data[3]])).to(device))

x_all = np.array(x_all)
y_all = np.array(y_all)
print(x_all.shape)
print(y_all.shape)

sampled_index = np.random.choice(np.array(list(range(len(x_all)))), size=1000, replace=False)
x_all=x_all[sampled_index]
y_all=y_all[sampled_index]
print(x_all.shape)
print(y_all.shape)

custom_dataset = DICOMDataset(x_all, y_all)

# Split dataset into train and test
train_size = int(0.8 * len(custom_dataset))
test_size = len(custom_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])

# Create dataloaders for train and test sets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


# Define model, loss function, and optimizer
model = SimpleCancerCNN()
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs=1000
# Train model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        inputs=torch.permute(inputs,(0,3,1,2))
        inputs=inputs.float()
        outputs = model(inputs)
        labels=labels.float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch %d, loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))


# Evaluate model on test set
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs=torch.permute(inputs,(0,3,1,2))
        inputs=inputs.float()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        print('Test loss: %.3f' % (loss / len(test_loader)))

torch.save(model,"./weights/B_"+str(num_epochs)+"_trained.pth")

#visualize predictions 
for i,data in enumerate(test_loader):
    if i==10:
        break
    else:
        #print(data[0][0].shape,data[1][0])
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        axes[0][0].imshow(data[0][0], cmap='gray')
        axes[0][0].set_title("original")
        mask = np.zeros_like(data[0][0])
        mask[data[1][0][1]:data[1][0][3], data[1][0][0]:data[1][0][2]] = 1
        axes[0][1].imshow(mask, cmap='gray')
        axes[0][1].set_title("original mask")
        inputs=torch.permute(data[0][0],(2,0,1))
        inputs=inputs.float()
        outputs = model(inputs).detach().numpy()[0]
        outputs=outputs.astype(int)
        mask_pred = np.zeros_like(data[0][0])
        mask_pred[outputs[1]:outputs[3], outputs[0]:outputs[2]] = 1
        axes[0][2].imshow(mask_pred, cmap='gray')
        axes[0][2].set_title("predicted mask")
        # Create a sample image
        image = data[0][0]
        image=image[outputs[1]:outputs[3], outputs[0]:outputs[2]]
        # Apply Sobel edge filter
        sobel_x = ndimage.sobel(image, axis=0)
        sobel_y = ndimage.sobel(image, axis=1)
        sobel = np.hypot(sobel_x, sobel_y)
        # Plot the original and filtered images
        axes[1][0].imshow(image, cmap='gray')
        axes[1][0].set_title('Point of interest predicted')
        axes[1][1].imshow(sobel, cmap='gray')
        axes[1][1].set_title('Sobel Filtered Image')
        fig.delaxes(axes[-1][-1])
        plt.savefig("./plots/train/"+str(i)+".png")
        