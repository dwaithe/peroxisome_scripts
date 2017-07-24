# peroxisome_scripts

This collection of jupyter notebooks and python functions are designed for handling multi-colour STED super-resolution images, aligning them and analysing them. For this work we have used confocal imaging to resolve specific regions of our input images and then analysed fluorescence distributions in one or more aligned STED channels. Because the STED channels are acquired in frame-mode there can be some drift between their acquistion when imaging more than one STED channel. In this situation we imaged confocal channels in parallel to the STED channels to a separate dectector. Because the confocal channel is identical in each case this can be aligned and can be used to correct for drift in the STED channels. The Double color analysis script includes functions calls which will align STED images based on confocal images acquired. Because the STED images and the confocal images may not be chromatically perfectly aligned we also include a function which can calculate from Tetraspeck beads the offset between the confocal and STED channels. Because we typically use a single donut for both STED channels, the chromatic correction between these two images is miminal.

- 'Single color analysis.ipynb': script for processing and analysing single channel STED data (+confocal). 
- 'Double color analysis.ipynb': script for processing and analysing double channel STED data (+confocal). 
- 'patch_fn.py': collection of functions used for the analysis.
- 'plot_data.ipynb': collection of functions for visualising and interpreting analysed data.

As a user you will either be interested in 'Single color analysis.ipynb' or 'Double color analysis.ipynb' depending on how many STED image channels you have. Within this script the processing and analysis is performed and the measurements saved to the computer. Using the 'plot_data.ipynb' you can then visualise distributions in the saved data.

The 'legacy' folder contain all the jupyter and Fiji/ImageJ scripts that were used for the original paper for which this analysis was developed:

Super resolution microscopy reveals compartmentalization of peroxisomal membrane proteins
Silvia Galiani, Dominic Waithe, Katharina Reglinski, Luis Daniel Cruz-Zaragoza, Esther Garcia, Mathias P. Clausen, Wolfgang Schliebs, Ralf Erdmann and Christian Eggeling.
(2016) doi: 10.1074/jbc.M116.734038

Since then the same analysis has been implemented in python alone, no longer requiring Fiji/ImageJ explicity. The Jupyter scripts are being actively supported in the jupyter scripts mentioned above and so we recommend users interested in the analysis use these scripts.

Legacy folder contents:
- 'bead_calibrate.ijm': script that measures the distance between beads to extract chromatic offset.
- 'run_batch.ijm': script that runs data through the various analysis.
- 'sing_ch_find_patches.ijm': script that allows extraction of regions from confocal channel punctua and also random regions in similar distribution.
- 'sing_ch_patches_to_stack.ijm': patch region measurement for single channel STED data.
- 'double_ch_patches_to_stac.ijm': patch region measurement for multi channel STED data.
- 'plot_data.ipynb': This version of the plot data works with the data output from the ImageJ/Fiji scripts.
