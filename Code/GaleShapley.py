

from collections import defaultdict
import numpy as np
import pandas
import os


def _make_dict(rep_pref, interval, side):
    
    current_dir = os.getcwd()
    os.chdir(rep_pref)
    
    PrefDict = {}
    
    if side == 'init':
    
        filename = 'pref_i_' + str(interval)
        
        if filename in os.listdir(rep_pref):
            
            PrefFrame_i = pandas.read_csv(filename, header='infer')
            
            header = PrefFrame_i.columns.values.tolist()
            
            for index, rows in PrefFrame_i.iterrows():
                
                PrefDict[rows[header[1]]] = rows.iloc[2:7].as_matrix()
                
    if side == 'after':
    
        filename = 'pref_f_' + str(interval)
        
        if filename in os.listdir(rep_pref):
            
            PrefFrame_f = pandas.read_csv(filename, header='infer')
            
            header = PrefFrame_f.columns.values.tolist()
            
            for index, rows in PrefFrame_f.iterrows():
                
                PrefDict[rows[header[1]]] = rows.iloc[2:7].as_matrix()
                
    return PrefDict


class Matcher:

    def __init__(self, men, women):
        '''
        Constructs a Matcher instance.
        Takes a dict of men's spousal preferences, `men`,
        and a dict of women's spousal preferences, `women`.
        '''
        self.M = men
        self.W = women
        self.wives = {}
        self.pairs = []

        # we index spousal preferences at initialization 
        # to avoid expensive lookups when matching
        
        self.mrank = defaultdict(dict)  # `mrank[m][w]` is m's ranking of w
        self.wrank = defaultdict(dict)  # `wrank[w][m]` is w's ranking of m

        for m, prefs in men.items():
            
            for i, w in enumerate(prefs):
                self.mrank[m][w] = i

        for w, prefs in women.items():
            for i, m in enumerate(prefs):
                self.wrank[w][m] = i
                                
    def __call__(self):
        return self.match()
    
    def prefers(self, w, m, h):
        '''Test whether w prefers m over h.'''        
        return self.wrank[w][m] < self.wrank[w][h]
    
    def after(self, m, w):
        '''Return the woman favored by m after w.'''
        i = self.mrank[m][w] + 1    # index of woman following w in list of prefs
        
        if i >= len(self.M[m]): return -1
        
        return self.M[m][i]
    
    def match(self, men=None, next=None, wives=None):
        '''
        Try to match all men with their next preferred spouse.
        
        '''
        
        if men is None: 
            men = list(self.M.keys())         # get the complete list of men
        if next is None: 
            # if not defined, map each man to their first preference
            next = dict((m, rank[0]) for m, rank in self.M.items())
            
        if wives is None: 
            wives = {}                  # mapping from women to current spouse
        
        if not len(men): 
            self.pairs = [(h, w) for w, h in wives.items()]
            self.wives = wives
            return wives
        
        m, men = men[0], men[1:]
        
        w = next[m]                     # next woman for m to propose to
        
        if w != -1:
                
            next[m] = self.after(m, w)      # woman after w in m's list of prefs
        
            if w in wives:
            
                h = wives[w]                # current husband
            
                if m in self.wrank[w]:      # test if m in list
                    
                    if h in self.wrank[w]:
            
                        if self.prefers(w, m, h):
                            men.append(h)           # husband becomes available again
                            wives[w] = m            # w becomes wife of m
                        else:
                            men.append(m)           # m remains unmarried
                            
                    else:
                        
                        men.append(h)
                    
                else:
                    men.append(m)
            else:
                wives[w] = m                # w becomes wife of m
                
            
        return self.match(men, next, wives)



def _get_linking(rep_pref, rep_link):
    
    os.makedirs(rep_link)
    
    IntervalList = []
    
    for filename in os.listdir(rep_pref):
        
        if 'i_' in filename:
        
            [text, number] = filename.split('i_')
        
            if int(number) not in IntervalList:
            
                IntervalList.append(int(number))
                
    for interval in IntervalList:
        
        _get_link(rep_pref, rep_link, interval)
        
    return


def _get_link(rep_pref, rep_link, interval):
    
    side = 'init'
    init = _make_dict(rep_pref, interval, side)

    side = 'after'
    after = _make_dict(rep_pref, interval, side)
    
    if len(init) <= len(after):
        
        m = init
        w = after
        short = 'init'
        
    else:
        
        w = init
        m = after
        short = 'after'
    

    match = Matcher(m, w)
    
    wives = match()
    
    PrefArray = np.zeros((len(wives), 3))
    i = 0

    for key, value in wives.items():
    
        #we invert the man-women order to fit
        
        if short == 'init':
    
            PrefArray[i,:] = [interval, value, key]
            i += 1
            
        if short == 'after':
            
            PrefArray[i,:] = [interval, key, value]
            i += 1
        
    PrefFrame = pandas.DataFrame(PrefArray, columns = 
                             ['interval', 'label time t', 'label time t+1'])
    
    save_name = 'linking_' + str(interval)

    
    os.chdir(rep_link)
    PrefFrame.to_csv(save_name, index = False)
    
    return

