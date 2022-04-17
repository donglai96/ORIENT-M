# -*- encoding: utf-8 -*-
'''
@File    :   load.py
@Time    :   2022/04/17 13:40:00
@Author  :   Donglai Ma
@Contact :   dma96@atmos.ucla.edu

Load the OMNI data using pyspedas
'''


import pyspedas
import pandas as pd
from pytplot import tplot
import numpy as np

def unix2datetime(unix_list):
    date_list = pd.to_datetime(unix_list, unit='s')
    return np.array(date_list)

def fill_gap(x, y, max_gap):
    """
    Find the gaps in the data and interpolate to fill them
    """

    dx = x[1:] - x[0:-1]  # time step
    dt0 = np.median(dx)

    # difference between dx and dt smaller than value
    if np.any(np.abs(dx - dt0) > 0.01 * dt0):  # check that time step is constant
        print("time step is not constant")  # time step should be constant; else something is wrong

    idx = np.where(np.isfinite(x) & np.isfinite(y))  # finding all valid values
    idx = idx[0]
   
    data_interp = np.interp(x, x[idx], y[idx], right=np.nan,
                            left=np.nan)  # Interpolate for all x values, including all gaps.
    
    # find gaps, and fill with nan if it is greater than max_gap
    for i in range(len(idx) - 1):  
        if (idx[i + 1] - idx[i]) > max_gap:
            
            data_interp[idx[i] + 1: idx[i + 1]] = np.nan 

    return data_interp

class input_var(object):
    def __init__(self,
                 name=None,

                 start_time='2012-10-01',
                 end_time='2017-10-01',
                 gap_max=180,
                 no_update = True,
                 data_type = '5min'
                 ):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.tdata = None
        self.data = None
        self.gap_max = gap_max
        self.no_update = no_update
        self.data_type = data_type
        self.vars = pyspedas.omni.data(trange=[self.start_time, self.end_time],datatype = self.data_type, notplot=True,
                                  varnames=self.name, no_update=self.no_update)


    def get_data(self, name,gap_fill=True):
        if name is 'f10_7':
            raise ValueError("Not available")


        self.tdata = self.vars[name]['x']
        self.data = self.vars[name]['y']

        if gap_fill:
            print('filling gap of %s' % self.name)
            self.data = fill_gap(self.tdata, self.data, max_gap=self.gap_max)
        self.tdate = unix2datetime(self.tdata)
        frame = pd.DataFrame(data = {'date':self.tdate,name:self.data})
        frame.set_index('date', inplace=True, drop=True)

        return frame