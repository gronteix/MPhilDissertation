from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

# Image processing tools
import skimage
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import skimage.measure
import skimage.feature
import scipy.ndimage
import skimage.data
from skimage import img_as_float
import scipy.misc

from PIL import Image


#enlever si ne marche pas
import imageio
import os

def _get_stack_grayscale(im_rep, threshold, gain, cells, 
                         background, min_size, min_dist, seg_type, rep, invert = True):

    i = 1

    os.makedirs(rep)
    current_dir = os.getcwd()
    os.chdir(im_rep)
    file_list  = sorted(os.listdir(im_rep))
    file_list.sort(key=len, reverse=False)
    
    #### Dichotomoy of cases
    
    if seg_type == 'naive':
        
        for filename in file_list:
        
            save_name = 'raw_' + str(i)
            
            sigmoid_image = _contrast_enhance(filename, im_rep, threshold, gain, invert)

            arr = _image_segment_none(sigmoid_image, cells, background, 
                                      min_size, min_dist)

            current_dir = os.getcwd()
            os.chdir(rep)
            np.save(save_name, arr)
            os.chdir(current_dir)
            i += 1
                
    if seg_type == 'distance':
        
        for filename in file_list:
        
            save_name = 'raw_' + str(i)
            
            sigmoid_image = _contrast_enhance(filename, im_rep, threshold, gain, invert)

            arr = _image_segment_distance(sigmoid_image, cells, background, 
                                          min_size, min_dist)

            current_dir = os.getcwd()
            os.chdir(rep)
            np.save(save_name, arr)
            i += 1
            os.chdir(current_dir)
    
    if seg_type == 'luminosity':
    
        for filename in file_list:
        
            save_name = 'raw_' + str(i)
            
            current_dir = os.getcwd()
            os.chdir(im_rep)
            file_list  = sorted(os.listdir(im_rep))
            file_list.sort(key=len, reverse=False)
            filename = file_list[i]

            raw_image = np.asarray(Image.open(filename).convert('L'))
            
            sigmoid_image = _contrast_enhance(filename, im_rep, threshold, gain, invert)

            arr = _image_segment_luminosity(sigmoid_image, raw_image, cells, 
                                            background, min_size, min_dist)

            current_dir = os.getcwd()
            os.chdir(rep)
            np.save(save_name, arr)
            os.chdir(current_dir)
            i += 1

    return


#### Basic building block functions ####

def _image_segment_none(sigmoid_image, cells, background, min_size, min_dist):
    
    basins = np.zeros_like(sigmoid_image)

    basins[sigmoid_image < background] = 1
    basins[sigmoid_image > cells] = 2
    basins = skimage.morphology.closing(basins)
    seg_test = basins.astype(int)
    seg_test = skimage.morphology.watershed(sigmoid_image, basins)
    seg_test = skimage.morphology.remove_small_objects(seg_test, min_size=min_size)
    
    final_seg = skimage.measure.label(seg_test, neighbors=None, background=None, 
                                    return_num=False, connectivity=1)
    
    final_seg = skimage.morphology.remove_small_objects(final_seg, min_size=min_size)

    return final_seg

def _image_segment_distance(sigmoid_image, cells, background, min_size, min_dist):
    
    basins = np.zeros_like(sigmoid_image)

    basins[sigmoid_image < background] = 1
    basins[sigmoid_image > cells] = 2
    basins = skimage.morphology.closing(basins)
    seg_test = basins.astype(int)
    seg_test = skimage.morphology.watershed(sigmoid_image, basins)
    seg_test = skimage.morphology.remove_small_objects(seg_test, min_size=min_size)
    
    labeled = skimage.measure.label(seg_test, neighbors=None, background=None, 
                                    return_num=False, connectivity=1)
    
    watershed1 = skimage.morphology.remove_small_objects(labeled, min_size=min_size)
    
    seg_clear = watershed1>1
    seg_clear = skimage.morphology.binary_opening(seg_clear)
    seg_clear = skimage.morphology.binary_closing(seg_clear)
    distances = scipy.ndimage.distance_transform_edt(seg_clear)
    local_max = skimage.feature.peak_local_max(distances,  min_distance=min_dist,
                            indices=False, footprint=None, labels = seg_clear)
    
    maxima = skimage.measure.label(local_max)
    final_seg = skimage.morphology.watershed(-distances, maxima, mask=seg_clear,
                                         watershed_line=False)
    
    final_seg = skimage.morphology.remove_small_objects(final_seg,
                                                        min_size=min_size)
    
    return final_seg

def _image_segment_luminosity(sigmoid_image, raw_image, cells, background, min_size, min_dist):
    
    basins = np.zeros_like(sigmoid_image)

    basins[sigmoid_image < background] = 1
    basins[sigmoid_image > cells] = 2
    basins = skimage.morphology.closing(basins)
    seg_test = basins.astype(int)
    seg_test = skimage.morphology.watershed(sigmoid_image, basins)
    seg_test = skimage.morphology.remove_small_objects(seg_test, min_size=min_size)
    
    labeled = skimage.measure.label(seg_test, neighbors=None, background=None, 
                                    return_num=False, connectivity=1)
    
    watershed1 = skimage.morphology.remove_small_objects(labeled, min_size=min_size)
    
    seg_clear = watershed1>1
    seg_clear = skimage.morphology.binary_opening(seg_clear)
    seg_clear = skimage.morphology.binary_closing(seg_clear)

    local_max = skimage.feature.peak_local_max(raw_image,  min_distance=min_dist,
                            indices=False, footprint=None, labels = seg_clear)
    
    maxima = skimage.measure.label(local_max)
    final_seg = skimage.morphology.watershed(-raw_image, maxima, mask=seg_clear,
                                         watershed_line=False)
    
    final_seg = skimage.morphology.remove_small_objects(final_seg,
                                                        min_size=min_size)
    
    return final_seg

def _contrast_enhance(filename, image_repository, threshold, gain, invert):
    
    #extracting the image from its original file
    current_dir = os.getcwd()
    os.chdir(image_repository)

    image_raw = np.asarray(Image.open(filename).convert('L'))
    
    #normalizing the image
    im = skimage.exposure.equalize_adapthist(image_raw)
    im_blur = (im - im.min()) / (im.max() - im.min())
    equalized = skimage.exposure.equalize_adapthist(im_blur)
    im = skimage.exposure.rescale_intensity(equalized, in_range='image', out_range='dtype')
    
    imsigmoid = skimage.exposure.adjust_sigmoid(im, cutoff = threshold, gain= gain, inv=False)#
    
    if invert == True:
        
        imsigmoid = np.ones_like(imsigmoid) - imsigmoid
    
    return imsigmoid

