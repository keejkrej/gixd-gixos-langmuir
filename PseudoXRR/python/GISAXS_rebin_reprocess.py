# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:17:34 2019
@author: shenc
"""
#%% import
import p08_general.P08ScanTools as P08ScanTools
import p08_general.fio_reader as fio_reader
import p08_GIXD.p08_GIXD as GIXD
import GIXD_additional_dependency as GIXD_add
import p08_GIXD.GIXD_plot as plt
import numpy as np
import copy
import time

#%% file import
path_raw = "T:/current/raw/"
path_exp = "T:/current/processed/GID/angularrebin/"
sample = "chamber_bkg"
scan = 1259

#%% geometry
"""
give the following:
    energy [eV], 
    incidence angle (alpha_i) [deg], 
    outgoing angle (alpha_f, or beta in LISA definition) [deg],
    detector-sample distance D_det [mm],
    horizontal pixel idx for primary beam px_h_0 (y for vertical mounting)
    vertical pixel idx for primary beam px_v_0 (x for vertical mounting)
    detector orientation = [-1, -1]: 1st -1: 0th pixel up; 2nd -1: 0th pixel at large stth
"""
D_ph = 0
D_ph_det = 560.7 - D_ph
px_h_0 = 1020
px_v_0 = 990
orient = [-1,-1]
print(orient)
rot3 = -0.6
det = 'eiger1m'
Yoneda_mode = "fix"
absQxy = False
axis_mode = 0   # axis = angle

#%% setup figure display
colorscale = [1,5000]
#%% default mask
mask = np.ones((1062, 1028))
#%% time start
start_time = time.time() 

#%% load data
fio_addr = path_raw + sample+"_"+"{:05d}".format(scan)+'.fio'        
fio_data = fio_reader.read(fio_addr)
frame_data = P08ScanTools.Scan()
frame_data.load_scan(fio_addr, auto_load_images = True)
det_label = "eiger"
frame_array = frame_data.image_data[det_label]
# load energy
energy = copy.deepcopy(fio_data[0]['energyfmb'])
# detector alpha_i and alpha_f angle
alpha_i = np.abs(2.*fio_data[0]['om'])
alpha_f = 0.0

#%% sum all up
print("summing all frames up")
img_array = np.sum(frame_array, axis = 0)            
# orient matrix according to the detector:
img_array = (copy.deepcopy(img_array)).T                

# prepare matrix: load lambda images into dictionary, convert pixel to q
img_data = {'chh': [], 'chv': [],'mat': [], 'detrot': []}
img_data['detrot'] = 0.0
#pyplot.imshow(img_array)
img_data['mat']= img_array  # vertical mounting
#pyplot.imshow(img_data['mat'])
img_data['chh'] = np.array([np.arange(0., img_data['mat'].shape[1])])
img_data['chv'] = np.arange(0, img_data['mat'].shape[0])
Itt=copy.deepcopy(GIXD.pixel2th(img_data, D_ph, D_ph_det, px_h_0, px_v_0, Yoneda_mode, energy, det, orient, alpha_f, rot3))  
#%% create mask
eiger1m_mask = np.where(Itt['mat']<4e9, 1, 0)

#%%
Itt_vertrebin = GIXD_add.ttrebin(Itt, mask = eiger1m_mask)
Itt_rebin = GIXD_add.tthrebin(Itt_vertrebin)

#%%
fig_Itt_rebin = plt.plot2d(Itt_rebin, axis = ['tth', 'tt'], range_I = colorscale)

#%% export plot and data
filename = "GID_" + sample+"_"+"{:05d}".format(scan)+'_angle'
plt.export(fig_Itt_rebin, filename, path_exp)
GIXD.export(Itt_rebin, filename, path_exp, axis = ['tth', 'tt'])