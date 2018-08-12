
import numpy as np
import pandas
import os
import numpy.random as rnd


def _cost_function_dist(prop1, prop2):

    [mass1, x1, y1] = prop1
    [mass2, x2, y2] = prop2

    return (x1 - x2)**2 + (y1 - y2)**2


def _make_EndsFrame(DataFrame):
    
    DataFrame = DataFrame.dropna(axis=0, how='any')
        
    un = DataFrame['label'].unique()
    arr = np.zeros((len(un), 3))
    i = 0
    
    for value in un:
    
        tmax = int(DataFrame.loc[DataFrame['label'] == value, 'frame'].max())
        tmin = int(DataFrame.loc[DataFrame['label'] == value, 'frame'].min())
    
        arr[i,:] = [value, tmin, tmax]
        i += 1
    
    return pandas.DataFrame(arr, columns = ['label', 'tmin', 'tmax'])
    

def _mass_merger(DataFrame, dist, pc):
    
    EndsFrame = _make_EndsFrame(DataFrame)
    
    while len(EndsFrame) > 0:
        
        print(len(EndsFrame))
        
        label = EndsFrame['label'].iloc[0]
        EndsFrame, DataFrame = _mass_merge(EndsFrame, DataFrame, label, dist, pc)
        
    return EndsFrame, DataFrame 



def _mass_merge(EndsFrame, DataFrame, label, dist, pc):
    
    tmax = EndsFrame.loc[EndsFrame['label'] == label, 'tmax'].iloc[0]
    
    frames = DataFrame['frame'].as_matrix()
    
    maxframe = np.amax(frames)
    
    if tmax == maxframe:
        
        index = EndsFrame.loc[EndsFrame['label'] == label].index[0]
        
        EndsFrame = EndsFrame.drop([index])
        
        return EndsFrame, DataFrame
    
    ParticleFrame = DataFrame.loc[(DataFrame['label'] == label) &
                            (DataFrame['frame'] == tmax)]
    
    xi = ParticleFrame['x'].iloc[0]
    yi = ParticleFrame['y'].iloc[0]
    massi = ParticleFrame['mass'].iloc[0]
    tmax = DataFrame.loc[DataFrame['label'] == label, 'frame'].max()
    
    ParticleFrame = DataFrame.loc[
            (DataFrame['frame'] == tmax +1) &
            (DataFrame['x'] < xi + dist) &
            (DataFrame['x'] > xi - dist) &
            (DataFrame['y'] < yi + dist) &
            (DataFrame['y'] > yi - dist)]
    
    for ind, rows in ParticleFrame.iterrows():
        
        initmass = DataFrame.loc[(DataFrame['label'] == rows['label']) &
                                (DataFrame['frame'] == tmax), 'mass']
        
        if len(initmass) > 0:
            
            initmass = initmass.iloc[0]
            
            if ((1 + pc)*rows['mass'] > initmass + massi > (1 - pc)*rows['mass']):
                
                LabelInit = label
                LabelLoc = rows['label']
                
                DataFrame = _merge_paths(DataFrame, LabelInit, LabelLoc)
                
                if len(EndsFrame.loc[EndsFrame['label'] == LabelLoc]) == 0:
                    index = EndsFrame.loc[EndsFrame['label'] == label].index[0]
        
                    EndsFrame = EndsFrame.drop(index)
        
                    return EndsFrame, DataFrame
                    
                
                index = EndsFrame.loc[EndsFrame['label'] == LabelLoc].index[0]
                EndsFrame.loc[EndsFrame['label'] == label, 'tmax'] = EndsFrame.loc[
                    EndsFrame['label'] == LabelLoc, 'tmax'].iloc[0]
                DataFrame.loc[DataFrame['label'] == LabelLoc, 'label'] = label
                EndsFrame = EndsFrame.drop(index)
                
                return EndsFrame, DataFrame
            
        
    index = EndsFrame.loc[EndsFrame['label'] == label].index[0]
        
    EndsFrame = EndsFrame.drop(index)
        
    return EndsFrame, DataFrame


def _merge_paths(DataFrame, LabelInit, LabelLoc):
    
    
    LocFrameInit = DataFrame.loc[DataFrame['label'] == LabelInit]
    tinitInit = LocFrameInit['frame'].min()
    tendInit = LocFrameInit['frame'].max()
    LocFrameLoc = DataFrame.loc[DataFrame['label'] == LabelLoc]
    tinitLoc = LocFrameLoc['frame'].min()
    tendLoc = LocFrameLoc['frame'].max()
    
    LocFrame = DataFrame.loc[(DataFrame['label'] == LabelLoc) &
                            (DataFrame['frame'] >=  max(tendInit, tendLoc)) &
                            (DataFrame['frame'] <= min(tendInit, tendLoc))]
    
    for index, rows in LocFrame.iterrows():
        
        xi = DataFrame.loc[(DataFrame['label'] == LabelInit) & 
                      (DataFrame['frame'] == rows['frame']), 'x'].iloc[0]
        
        yi = DataFrame.loc[(DataFrame['label'] == LabelInit) & 
                      (DataFrame['frame'] == rows['frame']), 'y'].iloc[0]
        
        massi = DataFrame.loc[(DataFrame['label'] == LabelInit) & 
                      (DataFrame['frame'] == rows['frame']), 'mass'].iloc[0]
            
        DataFrame.loc[(DataFrame['label'] == LabelInit) & 
                      (DataFrame['frame'] == rows['frame']), 'x'] = (xi + rows['x'])/2
        
        DataFrame.loc[(DataFrame['label'] == LabelInit) & 
                      (DataFrame['frame'] == rows['frame']), 'y'] = (yi + rows['y'])/2
            
        DataFrame.loc[(DataFrame['label'] == LabelInit) & 
                      (DataFrame['frame'] == rows['frame']), 'mass'] = (massi + rows['mass'])/2
        
        DataFrame.loc[(DataFrame['label'] == LabelInit) & 
                      (DataFrame['frame'] == rows['frame']), 'angle'] = np.arctan((yi - rows['y'])
                                                                                  /(xi - rows['x']))%np.pi
            
        DataFrame = DataFrame.drop(index)
        print(rows['particle'])
        print(rows['frame'])
    
    if tendLoc > tendInit:
        
        DataFrame.loc[DataFrame['label'] == LabelLoc, 'label'] = LabelInit
    
    return DataFrame




def _link_one(EndsFrame, DataFrame, label, memory, dist):
    
    print(len(EndsFrame))
    
    tmax = EndsFrame.loc[EndsFrame['label'] == label, 'tmax'].iloc[0]
    
    frames = DataFrame['frame'].as_matrix()
    
    maxframe = np.amax(frames)
    
    if tmax == maxframe:
        
        index = EndsFrame.loc[EndsFrame['label'] == label].index[0]
        
        EndsFrame = EndsFrame.drop([index])
        
        return EndsFrame, DataFrame
        
    LocFrame = EndsFrame.loc[(EndsFrame['tmin'] >= tmax) & 
                    (EndsFrame['tmin'] < tmax + memory +1)]
        
        
    # we retrieve the positions of the particle
        
    ParticleFrame = DataFrame.loc[(DataFrame['label'] == label) &
                            (DataFrame['frame'] == tmax)]
    
    xi = ParticleFrame['x'].iloc[0]
    yi = ParticleFrame['y'].iloc[0]
    massi = ParticleFrame['mass'].iloc[0]
    framei = ParticleFrame['frame'].iloc[0]
    propi = [massi, xi, yi]
    
    LocFrame['energy'] = 0
    
    ParticleFrame = DataFrame.loc[
            (DataFrame['frame'] > framei) &
            (DataFrame['frame'] < framei + memory + 1) &
            (DataFrame['x'] < xi + dist) &
            (DataFrame['x'] > xi - dist) &
            (DataFrame['y'] < yi + dist) &
            (DataFrame['y'] > yi - dist)]
        
    for ind, rows in LocFrame.iterrows():
        
        if rows['label'] not in ParticleFrame.label.values:
            
            LocFrame = LocFrame.drop([ind])
            
        if rows['label'] in ParticleFrame.label.values:
            
            looplabel = rows['label']
            looptime = rows['tmin']
        
            LocalParticleFrame = DataFrame.loc[
                (DataFrame['label'] == looplabel) &
                (DataFrame['frame'] == looptime)]
                
            x = LocalParticleFrame['x'].iloc[0]
            y = LocalParticleFrame['y'].iloc[0]
            mass = LocalParticleFrame['mass'].iloc[0]
            prop = [mass, x, y]
        
            LocFrame.loc[LocFrame['label'] == looplabel, 
                     'energy'] = _cost_function_dist(propi, prop)
            
    if len(LocFrame) == 0:
        
        index = EndsFrame.loc[EndsFrame['label'] == label].index[0]
        
        EndsFrame = EndsFrame.drop(index)
        
        return EndsFrame, DataFrame
    
    if len(LocFrame) > 0:
        
        cost_array = LocFrame['energy'].as_matrix()
        
        position = np.argmin(cost_array)
        
        if position <= len(cost_array):
        
            LinkLabel = LocFrame.iloc[position]['label']
                        
            index = EndsFrame.loc[EndsFrame['label'] == LinkLabel].index[0]
            EndsFrame.loc[EndsFrame['label'] == label, 'tmax'] = LocFrame.loc[
                LocFrame['label'] == LinkLabel, 'tmax'].iloc[0]
            DataFrame.loc[DataFrame['label'] == LinkLabel, 'label'] = label
            EndsFrame = EndsFrame.drop([index])
        
    return EndsFrame, DataFrame
        

def _link_ends(DataFrame, memory, dist):
    
    EndsFrame = _make_EndsFrame(DataFrame)
    
    while len(EndsFrame) > 0:
        
        label = EndsFrame['label'].iloc[0]
        EndsFrame, DataFrame = _link_one(EndsFrame, DataFrame, label, memory, dist)
        
    return EndsFrame, DataFrame


                        
                        
                    
            
            
