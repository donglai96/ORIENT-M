# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/04/17 13:57:52
@Author  :   Donglai Ma
@Contact :   dma96@atmos.ucla.edu
'''

from datetime import datetime,timedelta
import numpy as np
import os
from scipy.io import savemat,loadmat
import pandas as pd
import warnings

def datetime_to_datenum(date_time):
    """
    Convert Python datetime to Matlab datenum
    :param date_time: Datetime object
    :return: date_num in float
    """
    # change date to timestamp
    mdn = date_time + timedelta(days=366)
    frac = (date_time - datetime(date_time.year, date_time.month, date_time.day, 0, 0, 0)).seconds / (
            24.0 * 60.0 * 60.0)
    return mdn.toordinal() + frac


def pdframe_to_real(pdframe, real_dir, res='1min',format = 'pkl'):
    """
    Change pdframe to a mat file
    the pdframe is like:
                         proton_density  flowspeed  flow_pressue
date
2020-05-29 00:00:00             3.8      291.1      0.644018
2020-05-29 00:01:00             3.6      292.7      0.616848
2020-05-29 00:02:00             3.8      293.0      0.652452
2020-05-29 00:03:00             4.0      292.6      0.684918
2020-05-29 00:04:00             4.1      293.2      0.704923

    save the data to pickle file
    @param pdframe:pdframe
    @param matname:the target mat file name
    @param dir:directory
    @return:1
    """
    if format =='mat':
        tdata_datetime = pdframe.index.to_pydatetime()
        print(tdata_datetime[0])
        tdata = np.expand_dims(np.array([datetime_to_datenum(tdata_datetime[i])
                                         for i in range(0, len(tdata_datetime))]), axis=1)
        column_list = pdframe.columns.values

        if not os.path.exists(real_dir):
            os.makedirs(real_dir)
        for column_name in column_list:
            # The dir have a '/'
            file_name =real_dir + 'real_' + column_name + '_' + res + '.mat'
            data = np.expand_dims(pdframe[column_name].to_numpy(), axis=1)
            savemat(file_name, {'tdata': tdata, 'data': data})
    elif format =='pkl':
        column_list = pdframe.columns.values
        for column_name in column_list:
            file_name = real_dir + 'real_' + column_name + '_' + res + '.pkl'
            pdframe[column_name].to_pickle(file_name)


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
    #######################################
    # Do not interp the values at begin and end
    data_interp = np.interp(x, x[idx], y[idx], right=np.nan,
                            left=np.nan)  # Interpolate for all x values, including all gaps.
    # print(data_interp)
    # find gaps, and fill with nan if it is greater than max_gap
    for i in range(len(idx) - 1):  # loop through first to second to last element
        if (idx[i + 1] - idx[i]) > max_gap:
            ###################################
            data_interp[idx[i] + 1: idx[i + 1]] = np.nan 

    return data_interp

def frame_fillgap(pdframe, max_gap):
    pdframe_x = (pdframe.index.astype(np.int64) // 10**9).values
    column_name = pdframe.columns
    for name in column_name:
        pdframe_y = pdframe[name].values
        pdframe[name] = fill_gap(pdframe_x, pdframe_y, max_gap=max_gap)
    return pdframe


def unix2datetime(unix_list):
    date_list = pd.to_datetime(unix_list, unit='s')
    return np.array(date_list)

def warningtime(start_time, end_time, unix_time):

    if start_time < np.array(unix_time)[0]:
        warnings.warn('The datafile start_time is later than the output start_time, please update the data file')
    if end_time > np.array(unix_time)[-1]:
        warnings.warn('The end_time of datafile is earlier than the output end_time, this usually happens when dealing'
                      ' with real time data, if you are not calculating real time data, please update the data file')
def unix_time_serires_init(start_time,end_time,time_resolution = 5):
    time_delta =  pd.to_timedelta(time_resolution, unit='min').value / 1e9
    start_time_unix = pd.to_datetime(start_time).value / 1e9
    end_time_unix = pd.to_datetime(end_time).value / 1e9

    ntime = int(((end_time_unix - start_time_unix) / time_delta)) + 1
    time = np.linspace(start_time_unix, end_time_unix, ntime)
    return time