# macrophage cell segmentation

## Description
Implementations of a watershed (```watershed``` folder) and a CNN (```SegNet``` folder) based approach to segmenting macrophage cells in biomedical imagery.

## Runing and Implementation Details
Clone the repository and make sure all scripts are in your working directory (PATH) in MATLAB. Specify the directory of the source images by editing the corresponding code segment. 

### Watershed

The watershed approach first extracts candidate regions (```candidate_region_extraction.m```) containing one or more macrophage cells with overlapping cytoplasmic boundaries. These candidate regions are then isolated into individual images further segmented using a Canny edge detector (```edge_detection.m```). Place your input images into designated folders and first run the ```candidate_region_extraction.m``` script followed by ```edge_detection.m``` script.

### SegNet

The CNN approach uses SegNet with weights initialized from the VGG-16 network. In order to use VGG-16, you need to install Neural Network Toolbox™ Model for VGG-16 Network for MATLAB. After doing so, you need to label your training ground truth images so that pixels with value 1 correspond to the 'Background' pixels and pixels with value 2 correspond to the 'Cells'. Then simply run ```main.m```.

You can use the SegNet build for either candidate region detection or straight-up trying to detect cytoplasmic boundaries between overlapping cell regions.




