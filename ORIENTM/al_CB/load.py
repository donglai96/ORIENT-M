# @Author : Donglai
# @Time   : 1/4/2021 7:33 PM
# @Email  : dma96@atmos.ucla.edu donglaima96@gmail.com

import os
from datetime import datetime,timedelta
import pandas as pd
from .config import CONFIG
from contextlib import closing
import urllib.request as request
import shutil
from ..eflux.utils import warningtime
import numpy as np

def string2date(strlist):
    datelist = []
    for str in strlist:
        yr, doy, hr, min, sec = int(str[0:4]),int(str[5:8]),int(str[9:11]),int(str[12:14]),int(str[15:17])
        dt = datetime(yr - 1, 12, 31)
        delta = timedelta(days=doy, hours=hr, minutes=min, seconds=sec)
        datelist.append(dt + delta)
    return datelist

def parse(yr, doy, hr, min, sec):
    yr, doy, hr, min = [int(x) for x in [yr, doy, hr, min]]

    dt = datetime(yr - 1, 12, 31)
    delta = timedelta(days=doy, hours=hr, minutes=min, seconds=sec)
    return dt + delta

def load(start_time = None, end_time = None, no_update = False):
    """
    Load the al DATA from colorado's model
    @param start_time: datetime of start time
    @param end_time: datetime of end time
    @param no_update: whether update the data base
    @return:
    """

    if not os.path.exists(CONFIG['local_data_dir']):
        os.makedirs(CONFIG['local_data_dir'])

    if not os.path.exists(CONFIG['real_data_dir']):
        os.makedirs(CONFIG['real_data_dir'])

    # if time_now is None:
    #     date_now = datetime.utcnow()
    # else:
    #     date_now = time_now


    # Get the file list of month

    time_length = (end_time - start_time).days
    date_str_list = [(end_time - timedelta(days=x)).strftime("%Y_%m")
                     for x in range(time_length)]
    date_str_list.sort()
    month_str_list = list(dict.fromkeys(date_str_list))
    name_suffix = '.txt'

    filenames = [CONFIG['local_data_dir'] + 'al_' + x + name_suffix for x in month_str_list]
    urls = [CONFIG['remote_data_dir'] + 'al_' + x + name_suffix for x in month_str_list]
    if no_update is False:

        for url, file in zip(urls, filenames):
            print(url)

            with closing(request.urlopen(url)) as r:
                with open(file, 'wb') as f:
                    shutil.copyfileobj(r, f)
                    print('Downloaded  %s' % file)


    data_all = []
    colnames = ['unix_time','al']
    for file in filenames:
        df = pd.read_csv(file, header = None, sep = '\s+',skiprows=1,names=colnames)
        data_all.append(df)
    frame = pd.concat(data_all,axis = 0, ignore_index = True)

    # change the time to unix time

    time = string2date(frame['unix_time'].values)
    frame['unix_time'] = time
    warningtime(start_time,end_time,time)
    frame = frame.set_index('unix_time')
    frame = frame.loc[start_time:end_time]

        # fill the missing date with nan

    # find the missing date

    time_test = frame.index.values
    value_test = frame[colnames[-1]].values
    mingap = min(time_test[1:] - time_test[:-1])
    insert_index = np.where(time_test[1:] - time_test[:-1] > 288*mingap) # 18 here is 3 hour
    add_total = 0
    print(insert_index)
    for ind in insert_index[0]:
        
                            
        # generate the array
        ind = ind+add_total
        startvalue = time_test[ind]
        print(startvalue)
        endvalue = time_test[ind + 1]
        
        new_array = np.arange(startvalue+mingap,endvalue,mingap)
        #print(new_array)
        #insert the new array
        time_test = np.insert(time_test,ind+1,new_array)
        new_value_array = np.empty(len(new_array))
        new_value_array[:] = np.nan
        value_test = np.insert(value_test,ind+1,new_value_array)
        add_total += len(new_array)
    #create a new frame
    
    frame_new = pd.DataFrame(data={'unix_time':time_test,colnames[-1]:value_test})
    frame_new = frame_new.set_index('unix_time')
    return frame_new



