

from collections import defaultdict
import numpy as np
import pandas
import os



def _clean_frame(name, rep_link_prop, pathlength):
    
    os.chdir(rep_link_prop)
    df = pandas.read_csv(name, header='infer')
    #df = df.drop('Unnamed: 0', axis = 1)
    df = df.dropna(axis = 0, subset = ['label'])
    
    un = df['label'].unique()
    
    for value in un:
    
        if (len(df.loc[df['label'] == value]) < pathlength) & (len(df.loc[df['label'] == value]) > 0):
        
            df = df.drop(df[df['label'] == value].index)
            
    os.chdir(rep_link_prop)
    df.to_csv('PropertyFrameCleaned', index = False)
    
    return df

def _get_dynamic_prop(df):
    
    ResultFrame = pandas.DataFrame()
    
    un = df['label'].unique()
            
    for value in un:
        
        LocFrame = df.loc[df['label'] == value]
        
        LocFrame['dx'] = LocFrame['x'].shift(periods = 1,freq=None) - LocFrame['x']
        LocFrame['dy'] = LocFrame['y'].shift(periods = 1,freq=None) - LocFrame['y']
        
        LocFrame['dr'] = np.sqrt(LocFrame['dx']**2 + LocFrame['dy']**2)
        LocFrame['dt'] = -LocFrame['frame'].shift(periods = 1,freq=None) + LocFrame['frame']
        
        LocFrame['dr_stats'] = LocFrame['dr']/LocFrame['dt']
        
        LocFrame['ax'] = LocFrame['x'].shift(periods = 1,freq=None) + LocFrame['x'].shift(periods = -1,freq=None) - 2*LocFrame['x']
        LocFrame['ay'] = LocFrame['y'].shift(periods = 1,freq=None) + LocFrame['y'].shift(periods = -1,freq=None) - 2*LocFrame['y']
        
        LocFrame['phi'] = np.arctan2(LocFrame['dy'],LocFrame['dx'])
        
        ResultFrame = ResultFrame.append(LocFrame)
                
    return ResultFrame


def _get_dynamic_prop_old(df):
    
    un = df['label'].unique()
            
    for value in un:
    
        tinit = df.loc[df['label'] == value, 'frame'].min()
        tmax = df.loc[df['label'] == value, 'frame'].max()
   
        x0 = df.loc[(df['frame'] == tinit) & (df['label'] == value), 'x'].iloc[0]
        y0 = df.loc[(df['frame'] == tinit) & (df['label'] == value), 'y'].iloc[0]
        
        LocFrame = df.loc[df['label'] == value]
        
        for index, row in LocFrame.iterrows():
            
            if row['frame'] == tinit:
                
                tmemory = row['frame']
                xmemory = row['x']
                ymemory = row['y']
                thetamemory = row['theta']
                smemory = 0
                phimemory = 0
                vxmemory = 0
                vymemory = 0
                
                df.loc[(df['frame'] == tmemory) & (df['label'] == value), 'dr'] = 0
                df.loc[(df['frame'] == tmemory) & (df['label'] == value), 'dt'] = 0
                df.loc[(df['frame'] == tmemory) & (df['label'] == value), 't'] = 0
                df.loc[(df['frame'] == tmemory) & (df['label'] == value), 'isd'] = 0
                df.loc[(df['frame'] == tmemory) & (df['label'] == value), 's'] = 0
                df.loc[(df['frame'] == tmemory) & (df['label'] == value), 'dtheta'] = 0
                df.loc[(df['frame'] == tmemory) & (df['label'] == value), 'phi'] = 0
                df.loc[(df['frame'] == tmemory) & (df['label'] == value), 'dphi'] = 0
                df.loc[(df['frame'] == tmemory) & (df['label'] == value), 'phi_drift'] = 0
                df.loc[(df['frame'] ==tmemory) & (df['label'] == value), 'density'] = len(df.loc[
                    (df['frame'] == tmemory) & (df['x'] < xmemory + 80) & (df['y'] < ymemory + 80)
                    & (df['x'] > xmemory - 80) & (df['y'] > ymemory - 80)]) + 1
                
                df.loc[(df['frame'] == tmemory) & (df['label'] == value), 'ax'] = 0
                df.loc[(df['frame'] == tmemory) & (df['label'] == value), 'ay'] = 0
            
            if row['frame'] > tinit:
            
                t1 = tmemory
                t2 = row['frame']
            
                x1 = xmemory 
                x2 = row['x']
            
                y1 = ymemory
                y2 = row['y']
                
                dx = x2 - x1
                dy = y2 - y1
            
                deltar = np.sqrt((dx)**2 + (dy)**2)
                deltatime = t2 - t1
                
                df.loc[(df['frame'] == t2) & (df['label'] == value), 'dr'] = deltar
                df.loc[(df['frame'] == t2) & (df['label'] == value), 'dx'] = dx
                df.loc[(df['frame'] == t2) & (df['label'] == value), 'dy'] = dy
                df.loc[(df['frame'] == t2) & (df['label'] == value), 'dr_stats'] = deltar/deltatime
                df.loc[(df['frame'] == t2) & (df['label'] == value), 'dt'] = deltatime
                df.loc[(df['frame'] == t2) & (df['label'] == value), 'ax'] = (dx/deltatime - vxmemory)/deltatime
                df.loc[(df['frame'] == t2) & (df['label'] == value), 'ay'] = (dy/deltatime - vymemory)/deltatime
                df.loc[(df['frame'] == t2) & (df['label'] == value), 't'] = t2 - tinit
                df.loc[(df['frame'] == t2) & (df['label'] == value), 'isd'] = np.sqrt((x0-x2)**2 + (y0 - y2)**2)
                df.loc[(df['frame'] == t2) & (df['label'] == value), 's'] = deltar + smemory
                df.loc[(df['frame'] == t2) & (df['label'] == value), 'dtheta'] = row['angle'] - thetamemory
        
                if x1-x2 != 0:
            
                    df.loc[(df['frame'] == t2) & (df['label'] == value), 'phi'] = np.arctan2(dy,dx)
                    phi0 = df.loc[(df['label'] == value) & (df['phi'] != 0), 'phi'].iloc[0]
                    df.loc[(df['frame'] == t2) & (df['label'] == value), 'phi_drift'] = (np.arctan2(dy,dx) - phi0)
                    
                    if phimemory != 0:
                        
                        df.loc[(df['frame'] == t2) & (df['label'] == value), 'dphi'] = (np.arctan2(dy,dx) - phimemory)
        
                df.loc[(df['frame'] ==t2) & (df['label'] == value), 'density'] = len(df.loc[
                    (df['frame'] == t2) & (df['x'] < x1 + 80) & (df['y'] < y1 + 80)
                    & (df['x'] > x1 - 80) & (df['y'] > y1 - 80)]) + 1
            
                tmemory = t2
                xmemory = x2
                ymemory = y2
                thetamemory = row['angle']
                smemory = deltar + smemory
                phimemory = np.arctan2(dy,dx)
                vxmemory = dx/deltatime
                vymemory = dy/deltatime
                
    
    return df
