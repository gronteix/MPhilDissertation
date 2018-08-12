import numpy as np
import pandas
import os

import skimage

def _get_properties_stack(rep_data, rep_save, rep_raw, orientation_type = 'naive'):

    df = pandas.DataFrame(columns=['frame', 'particle', 'x', 'y', 'theta', 'a', 'b', 'mass'])

    os.makedirs(rep_save)

    os.chdir(rep_data)

    file_list  = sorted(os.listdir(rep_data))
    file_list.sort(key=len, reverse=False)

    for filename in file_list:

        [trash, start] = filename.split( '_' )
        [frame_no, trash] = start.split( '.' )

        df_temp = _get_properties(np.load(filename), int(frame_no), orientation_type, rep_raw)

        save_name = 'prop_' + frame_no
        current_dir = os.getcwd()
        os.chdir(rep_save)
        df_temp.to_csv(save_name, index = False)
        os.chdir(current_dir)

    return

def _get_properties(final_seg, frame_no, orientation_type, rep_raw):

    #récupération des propriétés sur chacune des particules sur un frame.

    un = np.unique(final_seg)
    un = un[un > 0]
    sorted(un)
    
    if len(un)>0:

        prop_array = np.zeros((np.amax(un), 8))
        i = 0

        for val in un:

            [xmf, ymf, orientation, a, b, masse] = _get_ind_properties(final_seg, val, orientation_type, rep_raw, frame_no)
            prop_array [i, :] = [frame_no, val, xmf, ymf, orientation, a, b, masse]
            i += 1

            columns = ['frame', 'particle', 'x', 'y', 'theta', 'a', 'b', 'mass']

            DataFrame_f = pandas.DataFrame(prop_array, columns=columns)

        return DataFrame_f.loc[(DataFrame_f['x'] > 0)]
    
    if len(un) == 0:
        
        columns = ['frame', 'particle', 'x', 'y', 'theta', 'a', 'b', 'mass']
        
        return pandas.DataFrame(columns=columns)

def _get_ind_properties(final_seg, val, orientation_type, rep_raw, frame_no):

    single_cell_array = skimage.img_as_uint(final_seg == val)

    single_cell_array = single_cell_array/np.amax(single_cell_array)

    if np.any(single_cell_array) > 0:
        
        if orientation_type == 'naive':

            y, x = np.nonzero(single_cell_array)

            mxf = np.mean(x)
            myf = np.mean(y)

            xmax = np.amax(x)
            xmin = np.amin(x)
            ymax = np.amax(y)
            ymin = np.amin(y)

            xloc = x - np.ones((1,len(x)))*mxf
            yloc = y - np.ones((1,len(x)))*myf
            
        if orientation_type == 'luminosity':
            
            current_dir = os.getcwd()
            os.chdir(rep_raw)
            file_list  = sorted(os.listdir(rep_raw))
            file_list.sort(key=len, reverse=False)
            filename = file_list[frame_no]
            image_raw = np.asarray(Image.open(filename).convert('L'))
            os.chdir(current_dir)
            
            mask = np.zeros_like(single_cell_array)
            masked = np.multiply(mask, image_raw)
            median = np.median(masked[masked >0])
            masked[masked < median] = 0
            
            y, x = np.nonzero(masked)
            
            mxf = np.mean(x)
            myf = np.mean(y)

            xmax = np.amax(x)
            xmin = np.amin(x)
            ymax = np.amax(y)
            ymin = np.amin(y)

            xloc = x - np.ones((1,len(x)))*mxf
            yloc = y - np.ones((1,len(x)))*myf

    if (x.any()) & (y.any()):

        theta, a, b = _cov_angle(xloc, yloc)
        masse = len(x)
        return mxf, myf, theta, a, b, masse

    return 0, 0, 0, 0, 0, 0

def _cov_angle(xloc, yloc):

    cov = np.cov(np.vstack([xloc, yloc]))
    evals, evecs = np.linalg.eig(cov)

    if np.abs(evals[0]) > np.abs(evals[1]):

        eignevec = evecs[0,:]
        a = np.abs(evals[0])
        b = np.abs(evals[1])

    else:

        eignevec = evecs[1,:]
        a = np.abs(evals[1])
        b = np.abs(evals[0])

    [x_v, y_v] = eignevec

    return np.arctan2(y_v, x_v), a, b
