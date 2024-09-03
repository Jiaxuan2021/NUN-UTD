import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import cv2
from PIL import Image
import os
from scipy.ndimage import label
from skimage.morphology import remove_small_objects

# Read the data
dataset_name = 'River_scene1'
NIR_path = fr'..\dataset\{dataset_name}\select_bands\NIR.tif'
GREEN_path = fr'..\dataset\{dataset_name}\select_bands\GREEN.tif'

parent_dir = os.path.dirname(NIR_path).split('\\')[2]

NIR = gdal.Open(NIR_path).ReadAsArray()
GREEN = gdal.Open(GREEN_path).ReadAsArray()
# calculate NDWI
sub = GREEN - NIR
sum = GREEN + NIR
NDWI = cv2.divide(sub, sum)

def remove_speckles(mask, min_size=100, connectivity=1):
    """
    Remove small speckles from a binary mask image.
    
    Parameters:
    - mask: numpy array, binary mask image where objects are 255 and background is 0.
    - min_size: int, the minimum size of objects to keep.
    - connectivity: int, connectivity (1 for 4-connectivity, 2 for 8-connectivity) for labeling.
    
    Returns:
    - numpy array, the processed mask with small speckles removed.
    """
    # Convert the mask to a boolean array (True for objects, False for background)
    binary_mask = mask == 0
    
    # Remove small objects
    cleaned_mask = remove_small_objects(binary_mask, min_size=min_size, connectivity=connectivity)
    
    # Convert boolean array back to original mask format (255 for objects, 0 for background)
    processed_mask = cleaned_mask.astype(np.uint8) * 255
    
    return processed_mask

mask = NDWI.copy()
threshold = 0.02
mask[mask > threshold] = 255
mask[mask <= threshold] = 0
processed_mask = remove_speckles(mask, min_size=100, connectivity=2)
plt.imshow(Image.fromarray(processed_mask.astype(np.uint8)))
plt.axis('off')
plt.show()

np.save(fr'./NDWI_{parent_dir}.npy', processed_mask)

# plot NDWI with different thresholds
fig = plt.figure(figsize=(15,6))
for i in range(5):
    water_mask = NDWI.copy()
    # set threshold
    water_threshold = (i + 1) * 0.05
    plt.subplot(1, 5, i + 1)
    plt.title('NDWI threshold = {:.2f}'.format(water_threshold))
    plt.axis('off')
    # create water mask
    water_mask[water_mask > water_threshold] = 255
    water_mask[water_mask <= water_threshold] = 0
    plt.imshow(Image.fromarray(water_mask.astype(np.uint8)))
plt.savefig(fr'./NDWI_{parent_dir}_test.png')

fig2 = plt.figure(figsize=(10,6))
water_mask2 = NDWI.copy()
# set threshold
water_threshold2 = 0.02
plt.title('NDWI threshold = {:.2f}'.format(water_threshold2))
plt.axis('off')
# create water mask
water_mask2[water_mask2 > water_threshold2] = 255
water_mask2[water_mask2 <= water_threshold2] = 0
plt.imshow(Image.fromarray(water_mask2.astype(np.uint8)))
plt.savefig(fr'./NDWI_{parent_dir}.png')

