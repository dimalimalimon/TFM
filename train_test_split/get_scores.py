import xml.etree.ElementTree as ET
import os
import sys

# Path to the directory containing the CT scan images and XML annotations
xml_dir = './Annotation/'+sys.argv[1]+"/"

# Parse the XML annotations
#xml_dir = os.path.join(data_dir, 'annotations')
xml_files = os.listdir(xml_dir)
print("Files:", xml_files)
scores = {}
for xml_file in xml_files:
    # Parse the XML file and extract the relevant information
    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()
    # Compute a score for each CT scan image based on the information in the XML file
    for image in root.findall('image'):
        image_id = image.get('id')
        score = 0
        for feature in image.findall('feature'):
            # Compute the score based on the proximity and size of the feature
            # You can customize this based on your specific use case
            score += feature.get('size') / feature.get('distance')
        scores[image_id] = score

# Rank the CT scan images based on their scores
sorted_images = sorted(scores, key=scores.get, reverse=True)
print(sorted_images)
