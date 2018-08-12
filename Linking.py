

from collections import defaultdict
import numpy as np
import pandas
import os


def _get_df(rep_prop):
    
    df = pandas.DataFrame()
    
    os.chdir(rep_prop)
    
    FileList = sorted(os.listdir(rep_prop))
    FileList.sort(key=len, reverse=False)
    
    for filename in FileList:
        
        [name, number] = filename.split('_')
                
        temp = pandas.read_csv(filename, encoding = 'utf-8')
        
        df.append(temp)
        
        df = pandas.concat([df, temp], ignore_index=True)
                
    return df


# In[31]:


def _init_label(df):
    
    i = 0
    cmin = df['frame'].min()
        
    for index, rows in df.iterrows():
                        
        if rows['frame'] == cmin:
            
            df.at[i,'label'] = rows['particle']
            
            i += 1
                            
    return df


# In[32]:


def _set_labels(df, LinkFrame):
        
    if len(LinkFrame) < 2: return df
    
    if len(df) < 2: return df
        
    tp = LinkFrame['interval'].iloc[0] + 1
    t = LinkFrame['interval'].iloc[0]
    
    i = 0
     
    for index, rows in LinkFrame.iterrows():
        
        particle_i = rows['label time t']
        particle_f = rows['label time t+1']
        
        label = df.loc[(df['particle'] == particle_i) &
                      (df['frame'] == t)]['label'].iloc[0]
                
        df.loc[(df['particle'] == particle_f) & 
               (df['frame'] == tp), 'label'] = label
    
    for index, rows in df.loc[df['frame'] == tp].iterrows():
                
        if not rows['label'].is_integer():
            
            particle_f = rows['particle']
            maxlabel = df['label'].max()
            
            df.loc[(df['particle'] == particle_f) & 
               (df['frame'] == tp), 
               'label'] = maxlabel + 1 
    
    return df


def _get_prop_frame(rep_prop, rep_link, rep_link_prop):
    
    df = _get_df(rep_prop)
    
    df = _init_label(df)
        
    os.chdir(rep_link)
    
    LinkList = sorted(os.listdir(rep_link))
    LinkList.sort(key=len, reverse=False)
    
    for linkfile in LinkList[:-1]:
        
        [name, number] = linkfile.split('_')
        
        os.chdir(rep_link)
        
        LinkFrame = pandas.read_csv(linkfile)
        
        df = _set_labels(df, LinkFrame)
               
        os.chdir(rep_link_prop)
        df.to_csv('PropertyFrame')
        
    return df

