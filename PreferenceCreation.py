
# coding: utf-8

# In[1]:


import numpy as np
import pandas
import os


# In[2]:


def _cost_function_mass(prop1, prop2):

    [mass1, x1, y1] = prop1
    [mass2, x2, y2] = prop2

    return np.log((x1 - x2)**2 + (y1 - y2)**2 + (mass1-mass2)**2 + 1)

def _cost_function_variable(prop1, prop2, a, b):

    [mass1, x1, y1] = prop1
    [mass2, x2, y2] = prop2

    return np.log(a*(x1 - x2)**2 + a*(y1 - y2)**2 + b*(mass1-mass2)**2 + 1)

def _cost_function_dist(prop1, prop2):

    [mass1, x1, y1] = prop1
    [mass2, x2, y2] = prop2

    return np.log((x1 - x2)**2 + (y1 - y2)**2 + 1)


# In[4]:


def _get_preferences_stack(r, minmass, cost, cost_type, rep_prop, rep_pref):

    os.makedirs(rep_pref)
    os.chdir(rep_prop)
    
    file_list = sorted(os.listdir(rep_prop))
    file_list.sort(key=len, reverse=False)
    
    for filename in file_list:
        
        [name, number] = filename.split('_')
        
        filename_p = name + '_' + str(int(number)+1)
                
        if filename_p in os.listdir(rep_prop):
        
            DataFrame_f = pandas.read_csv(filename_p, header='infer')
            DataFrame_i = pandas.read_csv(filename, header='infer')
            
            usef = DataFrame_f.loc[(DataFrame_f['mass'] > minmass)]
            usei = DataFrame_i.loc[(DataFrame_i['mass'] > minmass)]

            LinkFrame_i = _get_preferences(usei, usef, r, cost, cost_type)

            save_name = 'pref_i_' + number
            current_dir = os.getcwd()
            os.chdir(rep_pref)
            LinkFrame_i.to_csv(save_name, index = False)
            os.chdir(current_dir)
                       
            LinkFrame_f = _get_preferences(usef, usei, r, cost, cost_type)

            save_name = 'pref_f_' + number
            current_dir = os.getcwd()
            os.chdir(rep_pref)
            LinkFrame_i.to_csv(save_name, index = False)
            os.chdir(current_dir)

    return


# In[5]:


def _get_preferences(DataFrame_i, DataFrame_f, r, cost, cost_type):
    
    array = np.zeros((DataFrame_i.shape[0], 7))
    
    if (len(DataFrame_i) > 0) & (len(DataFrame_f) > 0):
    
        interval_i = DataFrame_i['frame'].as_matrix()[0]
    
        interval_f = DataFrame_f['frame'].as_matrix()[0]
    
        if interval_i < interval_f:
            start_name = 'particle ' + str(interval_i)
            finish_name1 = 'particle ' + str(interval_f) + ' choice 1'
            finish_name2 = 'particle ' + str(interval_f) + ' choice 2'
            finish_name3 = 'particle ' + str(interval_f) + ' choice 3'
            finish_name4 = 'particle ' + str(interval_f) + ' choice 4'
            finish_name5 = 'particle ' + str(interval_f) + ' choice 5'
            interval = interval_i
        
        if interval_i > interval_f:
            start_name = 'particle ' + str(interval_f)
            finish_name1 = 'particle ' + str(interval_i) + ' choice 1'
            finish_name2 = 'particle ' + str(interval_i) + ' choice 2'
            finish_name3 = 'particle ' + str(interval_i) + ' choice 3'
            finish_name4 = 'particle ' + str(interval_i) + ' choice 4'
            finish_name5 = 'particle ' + str(interval_i) + ' choice 5'
            interval = interval_f
    
        j = 0

        for index, rows in DataFrame_i.iterrows():

            label_i = [rows['particle']]
            label_f = _get_particle_preferences(rows, DataFrame_f, r, cost, cost_type)
            array[j] = np.concatenate(([interval], label_i, label_f), axis = 0)
            j += 1
    
    
        return pandas.DataFrame(array, columns = ['interval', start_name, finish_name1, 
                                                   finish_name2, finish_name3, finish_name4, 
                                                   finish_name5])
        
    return pandas.DataFrame(columns = ['interval', 'start_name', 'finish_name1', 
                                                   'finish_name2', 'finish_name3', 'finish_name4', 
                                                   'finish_name5'])


# In[6]:


def _get_particle_preferences(rows, DataFrame_f, r, cost, cost_type):

    [x1, y1] = [rows['x'], rows['y']]

    TestFrame = DataFrame_f.loc[(DataFrame_f['x'] < x1 + r) & (DataFrame_f['x'] > x1 - r)
                                & (DataFrame_f['y'] > y1 - r) & (DataFrame_f['y'] < y1 + r)]

    if TestFrame.empty: return -1, -1, -1, -1, -1

    prop1 = [rows['mass'], x1, y1]

    costarray = np.zeros((TestFrame.shape[0],2))
    i = 0
    
    for index, testrows in TestFrame.iterrows():

        prop2 = [testrows['mass'], testrows['x'], testrows['y']]
        
        if cost_type == 'mass':

            costarray[i,:] = [_cost_function_mass(prop1, prop2), index]
            i += 1
            
        if cost_type == 'distance':
            
            costarray[i,:] = [_cost_function_dist(prop1, prop2), index]
            i += 1
            
        if cost_type == 'variable':
            
            costarray[i,:] = [_cost_function_variable(prop1, prop2, a, b), index]
            i += 1
    
    index_array = -1*np.ones(5)
    mylist = costarray[:,0].tolist()
    
    for i in range(5):
    
        if len(mylist) > 0:
            
            if np.min(mylist) < cost:
                    
                ind = np.argmin(costarray[:,0])

                index_array[i] = TestFrame.loc[costarray[int(ind),1],'particle']
            
                costarray[ind, 0] = cost
            
                mylist.remove(min(mylist))

    return index_array

