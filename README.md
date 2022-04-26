# ORIENT-M
The Outer RadIation belt Electron Neural net model for Medium Energy electrons
The paper is submitted to space weather
The arxiv version:https://arxiv.org/abs/2109.08647
## Install

Notice for m1 mac:
since tensorflow support is not very well for now, this code won't work on m1 mac right now.


Download the repo, and 
```
python setup.py develop
```
install jupyter notebook to run the example
```
pip install jupyter notebook
```

## Example
The example is under Example folders

channel = [3,11,14,16] represents 54, 235, 597, 909 keV channels.
The model folder is under example Folder and also in zenodo.


```
import ORIENTM as orient


from datetime import datetime

start_time = datetime(2018,8,15)
end_time = datetime(2018,9,15)
input_time =  datetime(2018,9,1,3)

eflux_1 =  orient.eflux.model.ElectronFlux(start_time, end_time,instrument = 'mageis',channel = 11)
final_frame, X_input_total = eflux_1.get_flux(dst_source='omni',
                 al_source='al_CB',
                 sw_source='omni',use_omni = True,use_traj = False,get_input_time = input_time,get_MLT_flux=True,
                                             selected_MLT_datetime = input_time)
```
Example_storm shows a GEM event storm in June,2013
Example_asym shows an example of MLT dependence
Example_CuBoulder shows how to use AL index from CU Boulder instead of missing AL index in 2018
