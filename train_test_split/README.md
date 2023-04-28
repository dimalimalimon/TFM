##### Usage:

1. Create an anaconda environment:

`conda create -n dcm-vis python=3.7`

2. Installation:

`pip install -r requirements.txt`

3. Visualization:

For single: `python visualization.py --dicom-mode CT --dicom-path path/to/DICOM/folder --annotation-path path/to/ANNOTATION/file.xml --classfile category.txt`

For folder: `python visualization.py --dicom-mode CT --dicom-path path/to/DICOM/folder --annotation-path path/to/ANNOTATION/folder --classfile category.txt`

Press `ESC` to show next one


/home/dima/UOC/data/manifest-1608669183333/Lung-PET-CT-Dx/Lung_Dx-A0001/04-04-2007-NA-Chest-07990/2.000000-5mm-40805

/home/dima/UOC/TFM/Lung-PET-CT-Dx-Annotations-XML-Files-rev12222020/Annotation/A0001
python visualization.py --dicom-mode CT --dicom-path path/to/DICOM/folder --annotation-path path/to/ANNOTATION/folder --classfile category.txt

python visualization.py --dicom-mode CT --dicom-path /home/dima/UOC/data/manifest-1608669183333/Lung-PET-CT-Dx/Lung_Dx-A0001/ --annotation-path /home/dima/UOC/TFM/Lung-PET-CT-Dx-Annotations-XML-Files-rev12222020/Annotation/A0001 --classfile category.txt

