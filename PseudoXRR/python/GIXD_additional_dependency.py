#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: GIXD_additional_dependency.py
"""
Created on March, 2023
additional dependency
@author: Chen

"""
import p08_general.fio_reader as fio_reader
import numpy as np
from PIL import Image
import glob
import copy
import os.path as ospath


#%% rebin
def tthrebin(inputmap, dtth = 0.008, **kwargs):
    """
    rebin tth:
        with pixel splitting
    tth limit is the closest value from the limit in the original matrix
    input: n*m intensity matrix and tth matrix, tt-array
    argument: dtth = 0.008
    unit: degree
    output: n'*m' intensity matrix, m'-array for tth, n'-array for tt
    note: nan value for the tth region near specular
    note2: nan value for the tth/tt outside of the limit (limit either created or getting from the result of the ttrebin)
    do it by getting tt broder mask from ttrebin and multiply it to the mask, then temporarily set nan intensity to -1, in the end set everything outside the tth border to nan
    """
    # dtt = 0.008A^-1 at 15keV, tth = 0 for eiger at 600mm
    settings = { 'mask': [] }
    settings.update(kwargs)
    
    outputmap = copy.deepcopy(inputmap)
    #mask
    if settings['mask']==[]:
        mask = np.ones([inputmap['mat'].shape[0],inputmap['mat'].shape[1]])
        print('no mask')                
    else:
        mask = settings['mask']
        print('apply mask')
    
    #tt border mask from the tt-rebin: outside the border weight is zero
    if inputmap['ttborderMask']!=[]:
        mask = mask*inputmap['ttborderMask']
        print('apply tt border mask')
    
    # start
    outputmap = copy.deepcopy(inputmap)
    
    # getting tth limits of the input
    outputmap['tthlim_raw'] = np.zeros((inputmap['mat'].shape[0],2))
    for idx in range(outputmap['tthlim_raw'].shape[0]):
        finite_idx = np.where(np.isfinite(inputmap['mat'][idx,:]))
        outputmap['tthlim_raw'][idx,0] = np.min(inputmap['tth'][idx, finite_idx[0]])
        outputmap['tthlim_raw'][idx,1] = np.max(inputmap['tth'][idx, finite_idx[0]])
    
    # replace the nan into -1, their weight will be 0
    np.place(inputmap['mat'],np.isnan(inputmap['mat']), -1.0)
    np.place(outputmap['mat'],np.isnan(outputmap['mat']), 0.0)
                
    # GISAXS specular preparation: getting tth boundary around the specular
    tth_boundary = np.zeros((inputmap['mat'].shape[0],2))
    for idx in range(tth_boundary.shape[0]):
        pos_idx = np.where(inputmap['tth'][idx,:]>0)
        tth_boundary[idx,0] = np.min(inputmap['tth'][idx, pos_idx[0]])
        neg_idx = np.where(inputmap['tth'][idx,:]<0)
        if neg_idx[0].size != 0:
            tth_boundary[idx,1] = np.max(inputmap['tth'][idx, neg_idx[0]])
        else:
            tth_boundary[idx,1] = tth_boundary[idx,0] 
    
    # create rebinned tth array from the lower and the upper lim of the initial tth
    tthlim = [np.floor(np.min(inputmap['tth'])/dtth)*dtth, np.ceil(np.max(inputmap['tth'])/dtth)*dtth]
    outputmap['tth'] = np.array([np.arange(tthlim[0]-2*dtth, tthlim[1]+dtth*2, dtth)])
    # prepare for pixel splitting: idx, stat and the rebinned intensity matrix
    # idx :,:,0 is the index of the upper edge of the tth bin where each pixel belongs to; 
    # idx :,:,1 the fraction of the pixel into the upper edge of the tth bin, the rest goes to the lower edge
    outputmap['idx'] = np.zeros([inputmap['mat'].shape[0],inputmap['mat'].shape[1],2])
    # stat: statistics matrix for the rebinning: accumulated fraction into each rebined pixel, to be used for normalization, all values set to zero
    outputmap['stat'] = np.zeros([inputmap['mat'].shape[0],outputmap['tth'].shape[1]])
    # result matrix for intensity, all values set to 0
    outputmap['mat'] = np.zeros([inputmap['mat'].shape[0],outputmap['tth'].shape[1]])
    
    # pixel splitting:
    #   1. use np.digitize find the upper edge index in the rebinned matrix for each original pixel
    #   2. for each original pixel, compute its fraction for the upper edge index
    #   3. split the fraction of each pixel into the upper and the lower edge in the rebinning statistics matrix; the values in the rebinning statistics matrix starts accumulation
    #   4. split the intensity of each pixel into the upper and the lower edge in the rebinned matrix; the intensity in the rebinned matrix starts accumulation
    # prepare function
    create_digitize = np.digitize
    for i in range(inputmap['tth'].shape[0]):
        outputmap['idx'][i,:,0] = create_digitize(inputmap['tth'][i,:], outputmap['tth'][0,:], right = True)
        #print(outputmap['idx'][i,:,0])
        for j in range(inputmap['tth'].shape[1]):
            idx_bins = int(outputmap['idx'][i,j,0])
            #print([i, j, idx_bins])
            outputmap['idx'][i,j,1] = 1-(outputmap['tth'][0,idx_bins]-inputmap['tth'][i,j])/dtth
            outputmap['stat'][i,idx_bins] = outputmap['stat'][i,idx_bins] + outputmap['idx'][i,j,1]*mask[i,j]
            outputmap['stat'][i,idx_bins-1] = outputmap['stat'][i,idx_bins-1] + (1-outputmap['idx'][i,j,1])*mask[i,j]
            outputmap['mat'][i,idx_bins] = outputmap['mat'][i,idx_bins] + inputmap['mat'][i,j]*outputmap['idx'][i,j,1]*mask[i,j]
            outputmap['mat'][i,idx_bins-1] = outputmap['mat'][i,idx_bins-1] + inputmap['mat'][i,j]*(1-outputmap['idx'][i,j,1])*mask[i,j]
            #print([i, outputmap['stat'][i,213]])

    # prepare function
    create_argwhere = np.argwhere
    # normalization by the statistics, everything with no pixel in is counted as negative (-1e6)
    np.place(outputmap['stat'], outputmap['stat']==0, -1)    
    empty_cell = create_argwhere(outputmap['stat']<0)
    for y,x in empty_cell:
        outputmap['mat'][y,x] = 1e6

    outputmap['mat'] = outputmap['mat'] / outputmap['stat']    
    # test = copy.deepcopy(outputmap['mat'])
    # prepare function
    create_zeros = np.zeros
    create_interp = np.interp
    # interpolation for all these negative values
    for i in range(outputmap['mat'].shape[0]):
        empty_cell = create_argwhere(outputmap['mat'][i,:]<-1e3)[:,0]
        filled_cell = create_argwhere(outputmap['mat'][i,:]>-1e3-(1e-5))[:,0]
        filled_value = create_zeros(len(filled_cell))
        for j in range(len(filled_cell)):
            filled_value[j] = outputmap['mat'][i,filled_cell[j]]
        miss_values = create_interp(empty_cell, filled_cell, filled_value)        
        for j in range(len(empty_cell)):
            outputmap['mat'][i,empty_cell[j]]=miss_values[j]       
        # set mat element within tth boundary to nan for GISAXS
        #np.place(outputmap['mat'][i,:],(outputmap['tth'] - tth_boundary[i,0])*(outputmap['tth'] - tth_boundary[i,1])<0, np.nan)
        # set mat element beyond tth limit to nan
        np.place(outputmap['mat'][i,:],(outputmap['tth'] - outputmap['tthlim_raw'][i,0])*(outputmap['tth'] - outputmap['tthlim_raw'][i,1])>0, np.nan)
    
    del empty_cell, miss_values, filled_cell, filled_value, i, j 
    return outputmap

def ttrebin(inputmap, dtt = 0.008, **kwargs):
    """
    rebin tt, when tt is a n*m-matrix:
        with pixel splitting
    tt limit is the closest value from the limit in the original matrix
    input: n*m matrix for intensity, tth, tt
    argument: dtt = 0.008
    unit: A^-1
    output: n*m matrix for intensity and tth, n-array for tt
    works with the same mechanism as the tth rebin
    also output the tt limit into the output field and set the int value outside the tt border to nan for each column
    output also a tt border mask for the next step
    """
    settings = { 'mask': [] }
    settings.update(kwargs)
    
    outputmap = copy.deepcopy(inputmap)
    #mask
    if settings['mask']==[]:
        mask = np.ones([inputmap['mat'].shape[0],inputmap['mat'].shape[1]])
        print('no mask')                
    else:
        mask = settings['mask']
        print('apply mask')
    
    # getting tt limits of the original input
    outputmap['ttlim_raw'] = np.zeros((2,inputmap['mat'].shape[1]))
    for idx in range(outputmap['ttlim_raw'].shape[1]):
        unmask_idx = np.where(mask[:,idx]>0.5)
        if unmask_idx[0]!=[]:
            outputmap['ttlim_raw'][0,idx] = np.max(inputmap['tt'][unmask_idx[0],idx])
            outputmap['ttlim_raw'][1,idx] = np.min(inputmap['tt'][unmask_idx[0],idx])
        else:
            outputmap['ttlim_raw'][0,idx] = np.nan
            outputmap['ttlim_raw'][1,idx] = np.nan
    
    ttlim = [np.floor(np.min(inputmap['tt'])/dtt)*dtt, np.ceil(np.max(inputmap['tt'])/dtt)*dtt]
    #print('ttlim:\n%f\n%f\n' %(ttlim[0], ttlim[1]))
    outputmap['tt'] = np.arange(ttlim[0]-2*dtt, ttlim[1]+2*dtt, dtt)
    #print(len(outputmap['tt']))
    # prepare for pixel splitting: idx, stat and the rebinned intensity matrix
    # idx :,:,0 is the index of the upper edge of the tt bin where each pixel belongs to; 
    # idx :,:,1 the fraction of the pixel into the upper edge of the tt bin, the rest goes to the lower edge
    outputmap['idx'] = np.zeros([inputmap['mat'].shape[0],inputmap['mat'].shape[1],2])
    # stat: statistics matrix for the rebinning: accumulated fraction into each rebined pixel, to be used for normalization, all values set to zero
    outputmap['stat'] = np.zeros([outputmap['tt'].shape[0],inputmap['mat'].shape[1]])
    tth_stat = np.zeros([outputmap['tt'].shape[0],inputmap['mat'].shape[1]])
    # result matrix for intensity, all values set to 0
    outputmap['mat'] = np.zeros([outputmap['tt'].shape[0],inputmap['mat'].shape[1]])
    outputmap['tth'] = np.zeros([outputmap['tt'].shape[0],inputmap['mat'].shape[1]])    
    # pixel splitting:
    #   1. use np.digitize find the upper edge index in the rebinned matrix for each original pixel
    #   2. for each original pixel, compute its fraction for the upper edge index
    #   3. split the fraction of each pixel into the upper and the lower edge in the rebinning statistics matrix; the values in the rebinning statistics matrix starts accumulation
    #   4. split the intensity of each pixel into the upper and the lower edge in the rebinned matrix; the intensity in the rebinned matrix starts accumulation
    # prepare function
    create_digitize = np.digitize
    for i in range(inputmap['tt'].shape[1]):
        outputmap['idx'][:,i,0] = create_digitize(inputmap['tt'][:,i], outputmap['tt'], right = True)
        #print(outputmap['idx'][i,:,0])
        for j in range(inputmap['tt'].shape[0]):
            idx_bins = int(outputmap['idx'][j,i,0])
            #print([j, i, idx_bins])
            outputmap['idx'][j,i,1] = 1-(outputmap['tt'][idx_bins]-inputmap['tt'][j,i])/dtt
            outputmap['stat'][idx_bins,i] = outputmap['stat'][idx_bins,i] + outputmap['idx'][j,i,1]*mask[j,i]
            outputmap['stat'][idx_bins-1,i] = outputmap['stat'][idx_bins-1,i] + (1-outputmap['idx'][j,i,1])*mask[j,i]
            outputmap['mat'][idx_bins,i] = outputmap['mat'][idx_bins,i] + inputmap['mat'][j,i]*outputmap['idx'][j,i,1]*mask[j,i]
            outputmap['mat'][idx_bins-1,i] = outputmap['mat'][idx_bins-1,i] + inputmap['mat'][j,i]*(1-outputmap['idx'][j,i,1])*mask[j,i]
            tth_stat[idx_bins,i] = tth_stat[idx_bins,i] + outputmap['idx'][j,i,1]
            tth_stat[idx_bins-1,i] = tth_stat[idx_bins-1,i] + (1-outputmap['idx'][j,i,1])          
            outputmap['tth'][idx_bins,i] = outputmap['tth'][idx_bins,i] + inputmap['tth'][j,i]*outputmap['idx'][j,i,1]
            outputmap['tth'][idx_bins-1,i] = outputmap['tth'][idx_bins-1,i] + inputmap['tth'][j,i]*(1-outputmap['idx'][j,i,1])
            #print([i, outputmap['stat'][i,213]])
    
    # prepare function
    create_argwhere = np.argwhere
    # normalization by the statistics, everything with no pixel in is counted as negative (-1e6)
    np.place(outputmap['stat'], outputmap['stat']==0, -1)    
    np.place(tth_stat, tth_stat==0, -1)  
    empty_cell = create_argwhere(outputmap['stat']<0)
    for y,x in empty_cell:
        outputmap['mat'][y,x] = 1e6
    outputmap['mat'] = outputmap['mat'] / outputmap['stat']    
    outputmap['tth'] = outputmap['tth'] / tth_stat
    #test = copy.deepcopy(outputmap['mat'])
    
    # remove the columns without the values
    empty_col = []
    empty_row_limit = outputmap['mat'].shape[0]-2 
    for idx_value in range(outputmap['mat'].shape[1]):
        empty_cell = create_argwhere(outputmap['mat'][:,idx_value]<-1e3)[:,0]
        if len(empty_cell)>empty_row_limit:
            empty_col.append(idx_value)
    
    outputmap['mat'] = np.delete(outputmap['mat'], empty_col, axis = 1)
    outputmap['tth'] = np.delete(outputmap['tth'], empty_col, axis = 1)
    outputmap['ttlim_raw'] = np.delete(outputmap['ttlim_raw'], empty_col, axis = 1)
    #outputmap['stat'] = np.delete(outputmap['stat'], empty_col, axis = 1)
    #test = np.delete(test, empty_col, axis = 1)

    #empty_col_limt = outputmap['mat'].shape[1]*3/4
    empty_col_limt = outputmap['mat'].shape[1]-2
    # remove the 1st rows without the values
    for idx_value in range(outputmap['mat'].shape[0]):
        empty_cell = create_argwhere(outputmap['mat'][idx_value,:]<-1e3)[:,0]
        if len(empty_cell)<empty_col_limt:
            break
    #print(empty_cell)
    #print(idx_value)    
    outputmap['mat'] = np.delete(outputmap['mat'], range(0, idx_value+2), axis = 0)
    outputmap['tth'] = np.delete(outputmap['tth'], range(0, idx_value+2), axis = 0)
    outputmap['tt'] = np.delete(outputmap['tt'], range(0, idx_value+2), axis = 0)
    #outputmap['stat'] = np.delete(outputmap['stat'], range(0, idx_value+2), axis = 0)
    #test = np.delete(test, range(0, idx_value+2), axis = 0)
    
    # remove the last rows without the values
    idx_last = outputmap['mat'].shape[0]-1
    for idx_value in range(idx_last+1):
        empty_cell = create_argwhere(outputmap['mat'][idx_last-idx_value,:]<-1e3)[:,0]
        if len(empty_cell)<empty_col_limt:
            break
    #print(idx_last-idx_value-1)    
    outputmap['mat'] = np.delete(outputmap['mat'], range(idx_last-idx_value-1, idx_last+1), axis = 0)
    outputmap['tth'] = np.delete(outputmap['tth'], range(idx_last-idx_value-1, idx_last+1), axis = 0)
    outputmap['tt'] = np.delete(outputmap['tt'], range(idx_last-idx_value-1, idx_last+1), axis = 0)
    #outputmap['stat'] = np.delete(outputmap['stat'], range(idx_last-idx_value-1, idx_last+1), axis = 0)
    #test = np.delete(test, range(idx_last-idx_value-1, idx_last+1), axis = 0)
    
    # prepare function
    create_zeros = np.zeros
    create_interp = np.interp  
    # interpolation for all these negative values
    for i in range(outputmap['mat'].shape[1]):
        empty_cell = create_argwhere(outputmap['mat'][:,i]<-1e3)[:,0]
        filled_cell = create_argwhere(outputmap['mat'][:,i]>=-1e3-(1e-5))[:,0]
        filled_value = create_zeros(len(filled_cell))
        filled_tth = create_zeros(len(filled_cell))
        for j in range(len(filled_cell)):
            filled_value[j] = outputmap['mat'][filled_cell[j],i]
            filled_tth[j] = outputmap['tth'][filled_cell[j],i]
#        print(i)
#        print(empty_cell)
#        print(filled_cell)
#        print(filled_value)
#        if filled_cell.size:
#            miss_values = np.interp(empty_cell, filled_cell, filled_value)
#            miss_tth = np.interp(empty_cell, filled_cell, filled_tth)
#        else:
#            miss_values = np.zeros(len(empty_cell))
#            miss_tth = np.zeros(len(empty_cell))
#            for j in range(len(empty_cell)):
#                miss_values[j] = outputmap['mat'][empty_cell[j], i-1]
#                miss_tth[j]=outputmap['tth'][empty_cell[j], i-1]
        miss_values = create_interp(empty_cell, filled_cell, filled_value)
        #print(miss_values)
        miss_tth = create_interp(empty_cell, filled_cell, filled_tth)                
        for j in range(len(empty_cell)):
            outputmap['mat'][empty_cell[j],i]=miss_values[j]
            outputmap['tth'][empty_cell[j],i]=miss_tth[j]
        
        # set int value outside of tt limit raw to nan for every tth column
        if np.isnan(outputmap['ttlim_raw'][0,i]):
            outputmap['mat'][:,i] = np.nan
        else:
            np.place(outputmap['mat'][:,i],(outputmap['tt'] - outputmap['ttlim_raw'][0,i])*(outputmap['tt'] - outputmap['ttlim_raw'][1,i])>0, np.nan)
    
    # create a border mask: outside of the tt border the weight is zero
    outputmap['ttborderMask'] = np.ones([outputmap['mat'].shape[0],outputmap['mat'].shape[1]])
    nan_cell = create_argwhere(np.isnan(outputmap['mat']))
    for y,x in nan_cell:
        outputmap['ttborderMask'][y,x] = 0
    
    del empty_cell, miss_values, filled_cell, filled_value, miss_tth, filled_tth, nan_cell, i, j 
    return outputmap

