import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import os
import pandas as pd

directory = '/home/dima/UOC/data/manifest-1608669183333/Lung-PET-CT-Dx'
prefix="Lung_Dx-A"
matching_folders = [folder for folder in os.listdir(directory) if folder.startswith(prefix) and os.path.isdir(os.path.join(directory, folder))]

# Print the matching folders
print('Folders with prefix "{}":'.format(prefix))
#for folder in matching_folders:
 #   print(folder)
print("num of studies:",len(matching_folders))

training=matching_folders[:int(len(matching_folders)*0.7)]
test=matching_folders[int(len(matching_folders)*0.7):]

#print(training,"\n",test)

metadata=pd.read_csv("/home/dima/UOC/data/manifest-1608669183333/metadata.csv")
print(metadata)