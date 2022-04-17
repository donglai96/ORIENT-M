
from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
#from pandas._libs.tslibs import Hour

import tensorflow as tf
import os
from datetime import datetime, timedelta

from .config import CONFIG
from .. import ae_CB,al_CB
from .. import omni
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from .utils import *


class ElectronFlux(object):
    def __init__(self,
                 start_time,
                 end_time,
                 data_res = 5,
                 model_folder =CONFIG['model_dir'],
                 instrument = 'mageis',
                 channel = 11,
                 L_range = np.arange(2.6,6.6,0.1),
                 model_extra = None,
                 gap_max = 180


                 ):
        self.channel = channel
        self.start_time = start_time # datetime of start
        self.end_time = end_time # datetime of end, in prediction mode, it's the end of nowcast
        self.data_res = data_res
        self.model_folder = model_folder
        self.instrument = instrument
        # file name format is instrument_ch0_
        self.model_name = self.instrument + '_' +'ch' + str(self.channel)
        self.al = None
        self.ae = None
        self.dst = None
        self.flow_speed = None
        self.pressure = None
        self.bz = None

        self.model_extra = model_extra
        # For lag series
        if self.model_extra is None:
            # read the json format parameter 

            self._parameter = {}
            
            parameter_file = self.model_folder + '/' +self.model_name + '_parameter.txt'
            
            if (self.instrument =='mageis2')and (self.channel ==3):
                parameter_file = self.model_folder + '/' +'mageis_ch3' + '_parameter.txt'
            with open(parameter_file) as f:
                variables = json.load(f)
                for key, value in variables.items():

            
                    print(key,value)
                    self._parameter[key] = value

        # For input gap
        self.gap_max = gap_max
            

        
        # elif self.model_extra is '20days':
        #     self.startmin = -28800
        # self.endmin = 0
        # self.lagstep = 120
        self.version =self._parameter['version']
        self.startmin = self._parameter['startmin']
        self.endmin = self._parameter['endmin']
        self.lagstep = self._parameter['lagstep']
        self.input = self._parameter['input']
        

        # For sw data L1 point to earth
        # This is for dscover and ace data
        # 40 minutes
        self.timeshift = 40

        # For plot
        self.t_data = None
        self.L_range = L_range
        self.plot_matrix = None
        self.MLT_plot_matrix = None

        print('ORIENT-M initialized, Getting eflux from:',self.instrument,'channel:' ,self.channel)

    def get_flux(self,
                 dst_source = 'omni',
                 ae_source = 'ae_CB',
                 al_source = 'al_CB',
                 sw_source = 'omni',
                 makeplot = False,
                 no_update = False,
                 use_omni = False,
                 use_traj = False,
                 prediction_mode = 0,
                 prediction_length = 3,
                 prediction_source = 'dst',
                 # add MLT snap shot at selected time
                 get_MLT_flux = False,
                 selected_MLT_datetime = None, 
                 trajfolder = '/home/disk_f/data/eflux/selfmade/',
                 get_input_time = None):
        """
        Calculate the flux based on
        @param dst: dst or dst_kyoto
        @param ae: ae_CB or ae_Rice
        @param sw: dscover or ace
        @param selected_MLT_time: list of the selected MLT time if get_MLT_flux is True
        @return: the dataframe of flux
        """
        self.dst_source = dst_source
        self.ae_source = ae_source
        self.sw_source = sw_source
        self.al_source = al_source
        self.use_traj = use_traj
        self.trajfolder = trajfolder
        ## Add prediction
        # This part is for prediction mode
        # The input for prediction is :
        # # option 1:
        #  start_time, end_time, predict_endtime
        #  The model is same with before from start_time - end_time
        #  But from end_time to prediction_time:
        #  the model will use dst only and other input will be changed to average value 
        #  
        self.predict_mode = prediction_mode
        self.predict_length = prediction_length
        self.predict_end_time = self.end_time + timedelta(days = self.predict_length)
        self.predict_source = prediction_source # choose the input var for input
        # Load the data dst
        ## Maybe change end_time to data_end_time
        print("Start loading the data... np_update is:", no_update)
        if prediction_mode > 0:
            print("Prediction mode is on")
        if get_MLT_flux:
            print("Get MLT flux mode is on")
            if selected_MLT_datetime is None:
                print('No specific datetime for MLT selected, using end datetime!')

                # In case for the possible error, I want top dispace this following end time for MLT plot for 1 hour
                # But this issue could be resolved by turning prediction mode on

                self.MLT_datetime = self.end_time 
            else:
                self.MLT_datetime = selected_MLT_datetime
            # Check the MLT datetime
            if prediction_mode > 0 :
                if ((self.MLT_datetime < self.start_time) or  (self.MLT_datetime > self.predict_end_time)):
                    raise ValueError('Selected MLT time out of range!')
            else:
                if ((self.MLT_datetime < self.start_time) or (self.MLT_datetime > self.end_time)):
                    raise ValueError('Selected MLT time out of range!')




                #raise ValueError('No datetime for MLT_datetime has been selected! ')
        # if self.model_extra is None:
        #     self.data_start_time = self.start_time - timedelta(days = 12)
        # elif self.model_extra is '20days':
        #     self.data_start_time = self.start_time - timedelta(days = 22)
        self.data_start_time = self.start_time - timedelta(days = int(-self.startmin / 60/24)+2)

        



        if use_omni:
            self.omni_source = omni.load.input_var(name = ['AE_INDEX','AL_INDEX','SYM_H','flow_speed','Pressure','BZ_GSM'],
                                                   start_time= datetime.strftime(self.data_start_time,"%Y-%m-%d"),
                                                   end_time= datetime.strftime(self.end_time,"%Y-%m-%d"),
                                                   no_update = False,data_type='5min',gap_max=self.gap_max)

        if dst_source == 'dst':
            #self.dst = dst.load(self.data_start_time,self.end_time,no_update=no_update)
            raise ValueError("Dst source not found!Only support OMNI sym-h in this version!")
        elif dst_source =='omni':#dst_source =='dst_kyoto':
            self.dst = self.omni_source.get_data('SYM_H')
            #raise ValueError('dst source not found!')
        else:
            raise ValueError('dst source not found!')
        # Load the ae data
        if 'ae' in self.input:
            if ae_source == 'ae_CB':
                self.ae = ae_CB.load(self.data_start_time,self.end_time,no_update=no_update)
            # elif ae_source == 'ae_Rice':
            #     self.ae = ae_Rice.load(self.data_start_time,self.end_time,no_update=no_update)
                
            elif ae_source == 'omni':

                self.ae = self.omni_source.get_data('AE_INDEX')

            else:
                raise ValueError('ae source not found!')

        # Load the al data

        
        if 'al' in self.input:
            if al_source == 'al_CB':
                self.al = al_CB.load(self.data_start_time,self.end_time,no_update=no_update)
            # elif al_source == 'al_SuperMag':
            #     self.al = SuperMag.load(self.data_start_time,self.end_time)
            elif al_source =='omni':
                self.al = self.omni_source.get_data('AL_INDEX')
            else:
                raise ValueError('al source not found!')

        # Load the sw data

        # if sw_source == 'dscover':
        #     data_frame_sw  = dscover.load(self.data_start_time,self.end_time,no_update=no_update)
        #     self.flow_speed = data_frame_sw['Speed(km/s)'].copy().to_frame()
        #     self.pressure = data_frame_sw['Pressure(nPa)'].copy().to_frame()
        #     self.bz = data_frame_sw['Bz(nT)'].copy().to_frame()
        # elif sw_source =='ace':
        #     data_frame_sw  = ace.load(self.data_start_time,self.end_time)
        #     self.flow_speed = data_frame_sw['Speed(km/s)'].copy().to_frame()
        #     self.pressure = data_frame_sw['Pressure(nPa)'].copy().to_frame()
        #     self.bz = data_frame_sw['Bz(nT)'].copy().to_frame()
        
        if sw_source =='omni':
            self.flow_speed = self.omni_source.get_data('flow_speed')
            self.bz = self.omni_source.get_data('BZ_GSM')
            self.pressure = self.omni_source.get_data('Pressure')
            self.timeshift = 0
        else:
            raise ValueError('solar wind source not found')
        
        

        print("Loading data finished!")
    
        
        frame_name_list = self._parameter['input']
      ########################
        # prediction, if not source extend every frame with it's avg value
        # if source, reload with prediction value
        avg_value = {}
        # this avg_value is calculated from omni from 2012-10-01 to 2018-03-01

        avg_value['ae'] = 176.62
        avg_value['flow_speed'] = 423.38
        avg_value['pressure'] = 2.118
        avg_value['bz'] = - 0.044
        avg_value['al'] = -111.25
        
        if self.predict_mode>0:
            raise ValueError('Not supporting predict mode! Please contact the author for prediction version')
            self.split_time = self.end_time
            self.end_time = self.predict_end_time
            time_extend = timedelta(self.predict_length)
            for name in frame_name_list:
                print(name)
                if name != self.predict_source:
                    setattr(self,name,self.extend_frame(getattr(self,name),time_extend,mode = 'avg',value = avg_value[name]))
                else:
                    print('reloading : ', name)
                    self.dst = dst.load(self.data_start_time,self.end_time,no_update=no_update)
                    # right now only supports dst:
                    #########
                    # warning: this part of code should be utilized
                    
        #origin_frame_list = [self.ae,self.dst,self.pressure,self.flow_speed,self.bz]
        origin_frame_list = []
        for name in frame_name_list:
            origin_frame_list.append(getattr(self,name))
        
  


        # 1. Initialize the unix_time
        unix_time = unix_time_serires_init(self.start_time, self.end_time)

        tdate = unix2datetime(unix_time)

        # 2. Initialize the input matrix
        num_per_item = int((self.endmin - self.startmin) / self.lagstep)

        lag_matrix_all = np.zeros((len(unix_time),len(frame_name_list)*num_per_item))
        lag_name = []

        # 3. Fix the data and make item matrix
        # This part includes:
        # 1. get extra unix time (e.g 6 days data we need 16 days of the input)
        # 2. fill the gap of all the data and then interp the data on the unix time(16 days)
        # 3. Based on the lag make item matrix for each data

        unix_time_extra = np.arange(unix_time[0] + self.startmin * 60, unix_time[0], int(self.data_res * 60))
        unix_time_extend = np.append(unix_time_extra, unix_time)

        for frame,name,ii in zip(origin_frame_list, frame_name_list,range(len(frame_name_list))):
            # get the unix time
            unix_time_raw = frame.index.values.astype(np.int64)//1e9
            if name in ['bz','flow_speed','pressure']:
                unix_time_raw = unix_time_raw + 60 * self.timeshift 
                gap_max = 120
                if sw_source is 'ace':
                    gap_max = 1440
            else:
                gap_max = 10
            item_data_raw = fill_gap(unix_time_raw,frame.values.squeeze(),gap_max)

            # interp the data on unix_time_extend
            # frame_return is the frame for concatenate
            # interp_frame is the frame for rolling
            frame_return = pd.DataFrame(data = {'unix_time':unix_time})
            item_interp = np.interp(unix_time_extend, unix_time_raw,item_data_raw,left = np.nan,right = np.nan)
            interp_frame = pd.DataFrame(data = {'unix_time':unix_time_extend,
                                                'item': item_interp})
            lagnum = int(self.lagstep/self.data_res)
            item_avg = (pd.Series(item_interp).rolling(lagnum).mean()).values
            interp_frame['item_avg'] = item_avg

            for i in range(num_per_item):
                item_i = item_avg[(int((self.endmin-self.startmin) / self.data_res)
                                   - (1 + i * lagnum)):int(self.endmin / self.data_res) -(1 + i * lagnum)]
                name_t = name + '_t' + str(i)
                frame_return[name_t] = item_i

            # remove the unix_time for frame_return
            item_assem_frame = frame_return.iloc[:, 1:]

            lag_name += list(item_assem_frame.columns)
            lag_matrix_all[:, ii*num_per_item:(ii+1)*num_per_item] = item_assem_frame.values
        if self.instrument == 'rept':
            L_index = 2
            pos_matrix_all =  np.zeros((len(lag_matrix_all),4)) # time MLAT, MLT, R
        elif (self.instrument =='mageis') or (self.instrument =='mageis2'):
            L_index = 3
            pos_matrix_all =  np.zeros((len(lag_matrix_all),5))# time MLAT, sinMLT,cosMLT, R


        if self.use_traj:
            print('start reading traj info, using real traj, the folder is: ',self.trajfolder)
            time_unix_traj = pd.read_pickle(self.trajfolder + 'unix_time_5min')
            time_traj = pd.to_datetime(time_unix_traj,unit='s')

            # get the traj information(make a matrix) and cut the time

            data_traj_a = pd.DataFrame(data = {'time':time_traj})
            data_traj_b = pd.DataFrame(data = {'time':time_traj})



            position_name = ['Lm_eq', 'ED_MLT', 'ED_MLAT', 'ED_R']
            for name in position_name:
                file_name_a = 'rbspa' + '_' + name + '_OP77Q_intxt_5min'
                file_name_b = 'rbspb' + '_' + name + '_OP77Q_intxt_5min'
                data_traj_a[name] = pd.read_pickle(self.trajfolder + file_name_a)
                data_traj_b[name] = pd.read_pickle(self.trajfolder + file_name_b)

            print(data_traj_a)
            print(tdate)
            data_traj_a_frame = (data_traj_a[(data_traj_a['time']<=tdate[-1]) & ( data_traj_a['time']>=tdate[0]) ]).copy()

            data_traj_b_frame = (data_traj_b[(data_traj_b['time']<=tdate[-1]) & ( data_traj_b['time']>=tdate[0]) ]).copy()
            pos_matrix_traj_a = pos_matrix_all.copy()
            pos_matrix_traj_b = pos_matrix_all.copy()

            pos_matrix_traj_a[:,0] = unix_time
            pos_matrix_traj_a[:,2] = \
                np.sin(
                    np.deg2rad(data_traj_a_frame['ED_MLT'].values * 15)
                )
            pos_matrix_traj_a[:,3] = \
                np.cos(
                    np.deg2rad(data_traj_a_frame['ED_MLT'].values * 15)
                )
            pos_matrix_traj_a[:,1] =data_traj_a_frame['ED_MLAT'].values
            pos_matrix_traj_a[:,4] = data_traj_a_frame['ED_R'].values

            pos_matrix_traj_b[:, 0] = unix_time
            pos_matrix_traj_b[:, 2] = \
                np.sin(
                    np.deg2rad(data_traj_b_frame['ED_MLT'].values * 15)
                )
            pos_matrix_traj_b[:, 3] = \
                np.cos(
                    np.deg2rad(data_traj_b_frame[
                                   'ED_MLT'].values * 15)
                )
            pos_matrix_traj_b[:, 1] = data_traj_b_frame['ED_MLAT'].values
            pos_matrix_traj_b[:, 4] = data_traj_b_frame['ED_R'].values







        if not self.use_traj:
            pos_matrix_all[:,0] = unix_time
            total_matrix = np.concatenate((pos_matrix_all,lag_matrix_all),axis = 1)
        else:
            total_matrix_a =np.concatenate((pos_matrix_traj_a,lag_matrix_all),axis = 1)

            total_matrix_b = np.concatenate((pos_matrix_traj_b, lag_matrix_all), axis=1)





            # This is for traj

        # Get the model

        name_ = self.instrument + '_ch' + str(self.channel) + '_'
        if self.model_extra is not None:
            name_ = name_ + self.model_extra + '_'
        if (self.instrument =='mageis2') and (self.channel == 3):
            name_ = 'mageis_ch3_' 
        model_name = self.model_folder + name_ + 'model_sav.h5'
        model_avg = self.model_folder + name_ + 'input_avg.npy'
        model_std = self.model_folder + name_ + 'input_std.npy'
        nnmodel = tf.keras.models.load_model(model_name)
        eflux_avg = np.load(model_avg)
        eflux_std = np.load(model_std)
        #print(eflux_std[0:5])
        #print(eflux_avg[0:5])
        # remove nan here#################?

        if use_traj:
            X_input_a =    (total_matrix_a[:,1:] - eflux_avg)/eflux_std
            X_input_b = (total_matrix_b[:,1:] - eflux_avg)/eflux_std
            a_pred = nnmodel.predict(X_input_a).squeeze()
            b_pred = nnmodel.predict(X_input_b).squeeze()
            data_traj_a_frame['pred'] = a_pred
            data_traj_b_frame['pred'] = b_pred
            return data_traj_a_frame,data_traj_b_frame


        X_input = (total_matrix[:,1:] - eflux_avg)/eflux_std
        t_data = pd.to_datetime(total_matrix[:,0],unit  = 's')
        self.t_data = t_data
        L_data = self.L_range
        plot_matrix = np.zeros((len(t_data), len(L_data)))
        final_frame = pd.DataFrame(data={'date': t_data})


        for ll, i in zip(L_data, np.arange(len(L_data))):
            R_avg = eflux_avg[L_index]
            R_std = eflux_std[L_index]
            ll_norm = (ll - R_avg) / R_std
            X_input[:, L_index] = ll_norm
            y_pred = nnmodel.predict(X_input).squeeze()
            plot_matrix[:, i] = y_pred
            L_name = 'L_' + str(round(ll, 2))
            final_frame[L_name] = y_pred

        self.plot_matrix = plot_matrix

        # Get the MLT flux
        if get_MLT_flux:
            # Find the closest time in the dataset:
            MLT_time_index = self.t_data.get_loc(self.MLT_datetime, method = 'nearest')
            # Remember that use X_input as the base value, just need to change the position matrix
            MLT_base_X_input = X_input[MLT_time_index,:]

            # Now deal with the MLAT, sinMLT, cosMLT
            # 1. change MLAT (it's already been changed in the line 481)
            #  L_num : the grid num of L, 40 means (6.6 -2.6) /40 so it's 0.1 
            #  MLT_num : 36
            L_num = 40
            MLT_num = 36
            rmin = 2.6
            rmax = 6.6
            r = np.linspace(rmin, rmax, L_num)
            theta = np.linspace(0, 2 * np.pi, MLT_num)
            theta_mesh, r_mesh = np.meshgrid(theta, r)
            self.theta_mesh = theta_mesh
            self.r_mesh = r_mesh
            # avg and std, rename them incase get wrong order
            R_avg = eflux_avg[3] # notice the R is L at equator
            R_std = eflux_std[3]
            MLT_avg_sin = eflux_avg[1]
            MLT_std_sin = eflux_std[1]
            MLT_avg_cos = eflux_avg[2]
            MLT_std_cos = eflux_std[2]

            # This is right now designed for single time, it could be used as multi time or I could 
            # Create  similar to the plot I made before. But we don't need that much MLT data for now
            total_MLT_matrix = np.zeros((L_num,MLT_num,X_input.shape[1]))

            # tile the basement input
            total_MLT_matrix[:,:,:] = np.tile(MLT_base_X_input,(L_num, MLT_num, 1))

            # change L 

            ################# This part perhaps I should change to utils in the future
            for i in range(L_num):
                ll = rmin + i *(rmax - rmin)/L_num
                ll_norm = (ll - R_avg)/R_std
                total_MLT_matrix[i,:,3] = ll_norm

            for k in range(MLT_num):
                mlt = 0 + k*24 /MLT_num
                mlt_norm_sin = (np.sin(np.deg2rad(mlt*15)) - MLT_avg_sin)/MLT_std_sin
                mlt_norm_cos = (np.cos(np.deg2rad(mlt*15)) - MLT_avg_cos)/MLT_std_cos
                total_MLT_matrix[:,k,1] = mlt_norm_sin
                total_MLT_matrix[:,k,2] = mlt_norm_cos

            mlt_prediction = nnmodel.predict(total_MLT_matrix.reshape(-1,total_MLT_matrix.shape[-1]))
            print(mlt_prediction.shape)
            self.mlt_prediction_final = mlt_prediction.reshape(L_num, MLT_num)


            


        # Use final_frame to make plot
        if makeplot:
            self.make_plot()
            if get_MLT_flux:
                self.makeMLTplot()
        # get input for 
        if get_input_time is not None:
            self.get_input_time = get_input_time #list of time
            #create a input_dic for every input time
            
            input_time_index = self.t_data.get_loc(self.get_input_time, method = 'nearest')
            X_input_baseget = X_input[input_time_index,:].squeeze()
            
            
            
            L_num = 40
            # MLT_num = 36
            rmin = 2.6
            rmax = 6.6
            r = np.linspace(rmin, rmax, L_num)
            X_input_total = np.zeros((L_num, len(X_input_baseget)))
            for i in range(L_num):
                ll = rmin + i *(rmax - rmin)/L_num
                ll_norm = (ll - R_avg)/R_std
                X_input_total[i,:] = X_input_baseget
                X_input_total[i,3] = ll_norm
            #create the matrix for input
            # Now change the L_input
            
            return final_frame,X_input_total






        return final_frame

    def make_plot(self,normmax = 10**4):
        """
        Make the plot of prediction
        This function is used for plots on the equator
        @return:
        """
        t_data = self.t_data
        L_data = self.L_range
        time_mesh, L_mesh = np.meshgrid(t_data, L_data)
        fig,axs = plt.subplots(6,1,figsize = (16,16),sharex = True)
        norm = plt.Normalize(0, 7)
        ob = axs[5].pcolormesh(time_mesh, L_mesh, 10**self.plot_matrix.T, cmap='jet', norm=LogNorm(vmin=1, vmax=normmax)
                               )
        axs[5].set_ylabel('L_shell')
        color_axis = inset_axes(axs[1], width="1%",  # width = 5% of parent_bbox width
                                height="100%",
                                loc='lower left',
                                bbox_to_anchor=(1.05, 0., 1, 1),
                                bbox_transform=axs[5].transAxes,
                                borderpad=0,
                                )
        fig.colorbar(ob, cax=color_axis, label=r"$log_10$(flux)")
        # plot dst
        color = 'tab:red'
        axs[0].plot(self.dst.index,self.dst.values,color=color)
        axs[0].set_ylabel('symh from {}'.format(self.dst_source),color=color)
        axs[0].tick_params(axis='y', labelcolor=color)
        color = 'tab:blue'
        #plot flow speed
        axs[1].set_ylabel('flow_speed from {}'.format(self.sw_source),color=color)
        axs[1].plot(self.flow_speed.index,self.flow_speed.values,color=color)
        axs[1].tick_params(axis='y', labelcolor=color)


        # plot pressure
        axs[2].set_ylabel('pressure from {}'.format(self.sw_source),color=color)
        axs[2].plot(self.pressure.index,self.pressure.values,color=color)
        axs[2].tick_params(axis='y', labelcolor=color)
        axs[2].set_ylim(0,15)


        # plot bz
        axs[3].set_ylabel('bz from {}'.format(self.sw_source),color=color)
        axs[3].plot(self.bz.index,self.bz.values,color=color)
        axs[3].tick_params(axis='y', labelcolor=color)


        # plot ae or al
        if 'ae' in self.input:

            axs[4].set_ylabel('ae from {}'.format(self.ae_source),color=color)
            axs[4].plot(self.ae.index,self.ae.values,color=color)
            axs[4].tick_params(axis='y', labelcolor=color)
        elif 'al' in self.input:
            axs[4].set_ylabel('al from {}'.format(self.al_source),color=color)
            axs[4].plot(self.al.index,self.al.values,color=color)
            axs[4].tick_params(axis='y', labelcolor=color)

        # prediction mode
        if self.predict_mode> 0 :
            for ax in axs:
                yymin, yymax = ax.get_ylim()
                ax.vlines(self.split_time,yymin,yymax,colors = 'black',linestyles = 'dashed')


        if self.get_input_time is not None:
            for ax in axs:
                yymin, yymax = ax.get_ylim()
                ax.vlines(self.get_input_time,yymin,yymax,colors = 'black',linestyles = 'dashed')


        for ax in axs:
            ax.set_xlim(t_data[0],t_data[-1])

        plt.show()

    def makeMLTplot(self,normmax = 10**4):
        #plt.ioff()
        fig = plt.figure(figsize = (12,10))#set the size of subplots
        left,width=0.14,0.77
        bottom,height=0.11,0.5
        bottom_h=bottom+height+0.08
        rect_line1=[left,bottom,width,height]
        rect_line2=[left,bottom_h,width,0.2]
        
        #C = plot_matrix_total[tt,:,:].squeeze()
        plot_matrix_total = 10**self.mlt_prediction_final
        C = plot_matrix_total

            # setting up 'polar' projection and plotting

        axpolar=plt.axes(rect_line1,projection = 'polar')
        
        # axsymh=plt.axes(rect_line2)
        # axsymh.margins(x=0)
        # axsymh.plot(pd.to_datetime(time_omni,unit = 's'),symh,color = 'r',label = 'SYM-H')
        # axsymh.set_xlim(symhstart,symhend)

        def radian_function(x, y =None):
            rad_x = x/np.pi
            return "{}".format(str(int(12*rad_x)))
        # axsymh.plot(pd.to_datetime(time_omni,unit = 's'),AL,color = 'blue',label = '|AL|*0.1')
        
        
        
        # axsymh.set_ylabel('nT',fontsize = 16)
        # axsymh.legend()
        
        cs = axpolar.pcolor(self.theta_mesh, self.r_mesh, C,norm=LogNorm(vmin = 1,vmax=normmax),cmap = 'jet')
        axpolar.grid(True)
        
        axpolar.set_xticks(np.pi/180. * np.linspace(0,  360, 4, endpoint=False))
        axpolar.xaxis.set_major_formatter(ticker.FuncFormatter(radian_function))
        
        
        
        
        ##########
        #plot earth
        axpolar.set_yticks([2,4,6])
        rr = np.arange(0, 2, 0.01)
        theta_new = 2 * np.pi * rr
        axpolar.plot(theta_new,rr*0 + 1,'black')
        axpolar.set_ylim(0,6.5)
        r1 = np.arange(0,1,0.01)
        axpolar.plot(r1*0 + np.pi/2,r1,'black')
        axpolar.plot(r1*0 - np.pi/2,r1,'black')
        theta_new = np.arange(-0.5, 0.5, 1./180)*np.pi
        axpolar.fill_between(
        np.linspace(-np.pi/2, np.pi/2, 100),  # Go from 0 to pi/2
        0,                          # Fill from radius 0
        1,
        color = 'black'# To radius 1
        )       
    
        ################
        axpolar.tick_params(axis='x', labelsize=22)
        axpolar.tick_params(axis='y', labelsize=22)
        #axsymh.tick_params(axis='x', labelsize=12)
        #axsymh.tick_params(axis='y', labelsize=12)
        
        #axsymh.axvline(x=t_data[tt+timestart_index],color='black',linestyle="--")
        
        color_axis_1 = inset_axes(axpolar, width="5%",  # width = 5% of parent_bbox width
                                    height="100%",
                                    loc='lower left',
                                    bbox_to_anchor=(1.2, 0., 1, 1),
                                    bbox_transform=axpolar.transAxes,
                                    borderpad=0,
                                    )
        
        
        
        #########
        cb = fig.colorbar(cs,cax = color_axis_1)
        
        cb.ax.tick_params(labelsize=22) 
        cb.set_label(label=r'$cm^{-2}s^{-1}sr^{-1}keV^{-1}$',fontsize = 30)
        plt.show()
    
    

    @staticmethod
    def extend_frame(frame,time_extend,mode = 'avg',value = 0):
        """
        extend the frame with time_extend
        used for prediction
        """


        # get the time_resolution
        time_res = frame.index[-1] - frame.index[-2]
        # get the new frame
        extra_time_index = np.arange(frame.index[-1]+time_res,frame.index[-1]+time_extend,time_res)
        extra_value = np.repeat(value,len(extra_time_index))
        extra_Frame = pd.DataFrame(data = {frame.columns[0]:extra_value})
        extra_Frame.index = extra_time_index
        extra_Frame.index.name = frame.index.name
        new_Frame = frame.append(extra_Frame).copy()

        
        return new_Frame 
