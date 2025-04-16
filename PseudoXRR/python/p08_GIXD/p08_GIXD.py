#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: p08_GIXD.py
"""
Created on November, 2022
version 4.2
path /,  compatible in Windows and Linux
@author: Chen

processing rountine for GIXD data from P08
applied for Mythen-Soller or pinhole geometry
modification: qxy rebin is modified for GISAXS, with nan for specular plane, and nan outside of the measured region
deal with MG with small overlap (LISA style): change the decision of empty row and cell in qzrebin
mythen channel to tt allow fix Yoneda
"""
import p08_general.fio_reader as fio_reader
import numpy as np
#import fabio
from PIL import Image
import glob
import copy
import os.path as ospath

#%% for Mythen-Soller setup
def my2D(sample, scan, path_raw):
    """
    read the fio files and combine the mythen images into a 2D matrix
    sample as str
    scan as int
    path_raw as str, end with /
    """
    # create dictionary for the data
    mymap = {'tth': [], 'mych': [], 'mat': []}
    
    # load fio file
    fio_file = path_raw+sample+"_"+"{:05d}".format(scan)+".fio"
    flag_fio = True
    if ospath.isfile(fio_file):
        fio_data = fio_reader.read(fio_file)
        # load detrot/gid_det_tth as tth, into a numpy horizontal array (1, n)
        if "gid_det_tth" in fio_data[2]:
            mymap['tth'] = np.array([fio_data[2]['gid_det_tth']])
        else:
            mymap['tth'] = np.array([fio_data[2]['detrot']])
    else:
        fio_file = path_raw+sample+"_"+"{:05d}".format(scan)+".log"
        fio_data = fio_reader.read(fio_file)
        flag_fio = False
        scansetting = fio_data[3]['scan_cmd'].split(' ')
        tth_lim1 = float(scansetting[2])
        tth_lim2 = float(scansetting[3])
        tth_step = np.int(np.abs((tth_lim2-tth_lim1))/(float(scansetting[4])*float(scansetting[5])))
        mymap['tth'] = np.array([np.linspace(tth_lim1, tth_lim2, tth_step)])

    
    my_file = path_raw+sample+"_"+"{:05d}".format(scan)+"/mythen/"+sample+"_"+"{:05d}".format(scan)+"_0000_00.raw"
    mymap['mych'] = np.loadtxt(my_file)[:,0].astype(int)
    
    # load mythen data into a numpy matrix (1280, n)
    mymap['mat'] = np.zeros([mymap['mych'].shape[0],mymap['tth'].shape[1]])
    if flag_fio:
        for i in range(mymap['tth'].shape[1]):
            my_file = path_raw+sample+"_"+"{:05d}".format(scan)+"/mythen/"+sample+"_"+"{:05d}".format(scan)+"_"+"{:04d}".format(i)+"_00.raw"
            mymap['mat'][:,i] = np.loadtxt(my_file)[:,1]
    else:
        for i in range(mymap['tth'].shape[1]):
            my_file = path_raw+sample+"_"+"{:05d}".format(scan)+"/mythen/"+sample+"_"+"{:05d}".format(scan)+"_0000_"+"{:02d}".format(i)+".raw"
            mymap['mat'][:,i] = np.loadtxt(my_file)[:,1]
    
    del i, fio_file, my_file
    
    return mymap, fio_data

def GIXD_combine(inputmap_list, yaxis = 'mych'):
    """
    combine several GIXD scans together
    input: maps as list
    yaxis = 'mych', 'tt' or 'Qz'
    """
    outputmap = copy.deepcopy(inputmap_list[0])
    
    if yaxis == 'Qz':
        for i in range(len(inputmap_list)-1):
            outputmap['Qxy']=np.append(outputmap['Qxy'],inputmap_list[i+1]['Qxy'], axis = 1)
            outputmap['tth']=np.append(outputmap['tth'],inputmap_list[i+1]['tth'], axis = 1)
            outputmap['mat']=np.append(outputmap['mat'],inputmap_list[i+1]['mat'], axis = 1)    
    else:
        #print(len(args)-1)
        for i in range(len(inputmap_list)-1):
            #print(i)
            outputmap['tth']=np.append(outputmap['tth'],inputmap_list[i+1]['tth'], axis = 1)
            outputmap['mat']=np.append(outputmap['mat'],inputmap_list[i+1]['mat'], axis = 1)

    if outputmap[yaxis].ndim > 1:
        for i in range(len(inputmap_list)-1):
            outputmap[yaxis]=np.append(outputmap[yaxis],inputmap_list[i+1][yaxis], axis = 1)

    return outputmap

def mych2tt(inputmap, Ddet, energy=15000, alpha_i=0.07, myorient = 1, mych_0_est = 26, Yoneda_mode = "fit"):
    """
    convert the mythen channels into tt
    Ddet    [mm]
    energy  [eV]
    alpha_i [deg]
    myorient 1 0th channel down, -1 0th channel up
    output: n*m matrix for intensity, n-vector-v for tt, m-vector-h for tth
    """
    # constant preparation
    planck = 12398.4
    wv = planck / energy
    qc = 0.0216
    my_pixel = 0.05
    
    # build up structure
    outputmap = copy.deepcopy(inputmap)
    
    # orient the matrix and mythen channel array
    if myorient == 1:
        print('0th Mythen channel is down')
    elif myorient == -1:
        #flip the matrix and the mych array
        outputmap['mat']=np.flip(inputmap['mat'],0)
        outputmap['mych']=np.flip(inputmap['mych'],0)
        print('0th Mythen channel is up')
    else:
        print('interrupted proccess!\nchoose between:\nmyorient = 1: 0th Myth channel down\nmyorient = -1: 0th Myth channel up')
    
    # find Vineyard peak, idx_V is the index of the peak in the mych array
    if Yoneda_mode == "fit":
        mych_V = outputmap['mych'][np.argmax(np.sum(outputmap['mat'], axis = 1)[0:200])]
        print(mych_V)
        mych_0 = mych_V - myorient * (Ddet*np.tan(np.arcsin(qc*wv/4./np.pi))/my_pixel)
    else:
        mych_0 = mych_0_est
        mych_V = int(mych_0 + myorient * (Ddet*np.tan(np.arcsin(qc*wv/4./np.pi))/my_pixel))
                   
    outputmap['tt'] = np.degrees( np.arctan((outputmap['mych'] - mych_0)*myorient * my_pixel/Ddet) )
    outputmap['mych_V'] = mych_V
    
    del inputmap
    return outputmap

#%% pinhole setup with p100k
def p100k2D(sample, scan, path_raw, orient = 'vertical', mask = np.ones([195, 487])):
    """
    read the fio files and sum the detector images into a 2D matrix
    sample as str
    scan as int
    path_raw as str, end with /
    orient as str "vertical" or "horizontal"
    mask = matrix with 0 for masked points
    output: n*m matrix for intensity, n-vector-v for chv, m-vector-h for chh
    """
    # create dictionary for the data
    p100kmap = {'chh': [], 'chv': [],'mat': [], 'detrot': []}
    
    # load fio file
    fio_file = path_raw+sample+"_"+"{:05d}".format(scan)+".fio"
    fio_data = fio_reader.read(fio_file)

    img_num = len(fio_data[2]['p100k_roi1'])
    if "gid_det_tth" in fio_data[0]:
        p100kmap['detrot'] = fio_data[0]['gid_det_tth']
    else:
        p100kmap['detrot'] = fio_data[0]['detrot']
    
    img_addr = path_raw + sample+"_"+"{:05d}".format(scan)+"/p100k/"+"*.tif"
    img_files = glob.glob(img_addr)
    #print(img_files)
        
    # open data, 

    p100kmap['mat'] = np.zeros([195, 487])
    for i in range(img_num):
        #p100kmap['mat'] = p100kmap['mat']+fabio.open(img_files[i]).data
        p100kmap['mat'] = p100kmap['mat']+np.array(Image.open(img_files[i]))

    p100kmap['mat'] = p100kmap['mat']*mask        
        
    if orient == 'vertical':
        p100kmap['mat'] = p100kmap['mat'].T
            
    p100kmap['chh'] = np.array([np.arange(0., p100kmap['mat'].shape[1])])
    p100kmap['chv'] = np.arange(0, p100kmap['mat'].shape[0])
    
    return p100kmap, fio_data

def pixel2th(inputmap, D_ph, D_ph_det, px_h_0, px_v_0_est, v_0_mode = "fit", energy = 15000, det = 'p100k', orient = [-1, -1], D_rot2 = 0, D_rot3 = 0, **kwargs):
    """
    convert pixel into tth and tt
    D_ph    [mm]
    D_ph_det [mm]
    center_h
    center_v_est
    v_0_mode = "fix" or "fit"
    det = 'p100k', or lambda, eiger, perk
    orient = [-1, -1] (0,0 pixel up, larger detrot (pos>neg)), [0] for up/down, [1] for detrot direction
    D_rot2 = rotation2 [deg] detector 2theta angle from the horizontal plane (rot2 is pyFAI definition)
    D_rot3 = rotation3 [deg]  detector rotation arount the incident beam when at detrot=0, clockwise for rot3 up
    optional: mask
    output: n*m matrix for intensity, qxy and qz
    """
    # constant preparation
    planck = 12398.4
    wv = planck / energy
    qc = 0.0216
            
    px_size_pool = {'p100k': 0.172, 'lambda': 0.055, 'eiger1m': 0.075, 'perk': 0.200}
    px_size = px_size_pool[det]
    D_rot2_rad = np.radians(D_rot2)
    D_rot3_rad = np.radians(D_rot3)
    inputmap['rotmat'] = np.array([[np.cos(-D_rot3_rad), -np.sin(-D_rot3_rad)],[np.sin(-D_rot3_rad), np.cos(-D_rot3_rad)]])
    
    # find Vineyard peak, idx_V is the index of the peak in the chv array
    if v_0_mode == "fit":
        px_v_V = inputmap['chv'][np.argmax(np.sum(inputmap['mat'], axis = 1)[px_v_0_est-20:px_v_0_est+20])+px_v_0_est-20]
        #print(px_v_V)
        px_v_0 = px_v_V - orient[0] * ((D_ph_det+D_ph)*np.tan(np.arcsin(qc*wv/4./np.pi) - D_rot2_rad)/px_size)
        #print(px_v_0)
    else:
        px_v_0 = px_v_0_est
        px_v_V = int(px_v_0 + orient[0] * ((D_ph_det+D_ph)*np.tan(np.arcsin(qc*wv/4./np.pi) - D_rot2_rad)/px_size))
        
    # matrix for horizontal and vertical distance of a pixel to the beam center, in the detector frame: :,:,0 for horizontal, :,:,1 for vertical 
    px_pos_det = np.zeros([inputmap['mat'].shape[0],inputmap['mat'].shape[1],2])
    
    for i in range(px_pos_det.shape[0]):
        px_pos_det[i,:,0] = orient[1] * px_size*(inputmap['chh']-px_h_0)
    for i in range(px_pos_det.shape[1]):
        px_pos_det[:,i,1] = orient[0] * px_size*(inputmap['chv']-px_v_0)
    
    #calculate the coordinate of the pixels at the frame of the vertical slit, for rotated detector
    px_pos = np.zeros([inputmap['mat'].shape[0],inputmap['mat'].shape[1],2])

    px_pos[:,:,0] = inputmap['rotmat'][0,0]*px_pos_det[:,:,0] + inputmap['rotmat'][0,1]*px_pos_det[:,:,1]   # horizontal pos
    px_pos[:,:,1] = inputmap['rotmat'][1,0]*px_pos_det[:,:,0] + inputmap['rotmat'][1,1]*px_pos_det[:,:,1]   # vertical pos
    
    # distance projection of the pixels at the same height position as the poni pixel from the pinhole position
    D_px = np.zeros([inputmap['mat'].shape[0],inputmap['mat'].shape[1],2])
    # projection on the horizontal plane
    D_px[:,:,0] = (D_ph + D_ph_det)*np.cos(D_rot2_rad) - px_pos[:,:,1]*np.sin(D_rot2_rad) - D_ph
    # height of the pixel from the horizontal plane
    D_px[:,:,1] = (D_ph + D_ph_det)*np.sin(D_rot2_rad) + px_pos[:,:,1]*np.cos(D_rot2_rad)
    # calculate the tth and tt based on the distance projections
    inputmap['tth']=np.degrees(np.arctan(px_pos[:,:,0]/D_px[:,:,0]))+inputmap['detrot'];
    inputmap['tth'] = np.where(np.abs(inputmap['tth'])<1e-8, 1e-8, inputmap['tth'])
    tt_rad=np.arctan(D_px[:,:,1]/(D_ph*np.sin(np.radians(inputmap['detrot']))/np.sin(np.radians(inputmap['tth'])) + D_px[:,:,0]/np.cos(np.radians(inputmap['tth']-inputmap['detrot']))));

    inputmap['tt'] = np.degrees(tt_rad)
    inputmap['px_v_V'] = px_v_V
    
    return inputmap

#%% angle-Q convertion
def th2q(inputmap, energy=15000, alpha_i=0.07, absQxy = True):
    """
    convert the tt and tth into Q
    energy  [eV]
    alpha_i [deg]
    output: n*m matrix for intensity, 
            for mythen:n*m matrix for qz and qxy
            for pinhole: n*m matrix for qxy and qz
    """
    # constant preparation
    planck = 12398.4
    wv = planck / energy
    k_i = 2*np.pi/wv
    
    # build up structure
    outputmap = copy.deepcopy(inputmap)    
    del inputmap    
    # Qz matrix and Qxy matrix
    #Qz = k_i * (np.sin(np.radians(outputmap['tt'])) + np.sin(np.radians(alpha_i)))
    outputmap['Qz'] = np.zeros([outputmap['mat'].shape[0],outputmap['mat'].shape[1]])
    outputmap['Qxy'] = np.zeros([outputmap['mat'].shape[0],outputmap['mat'].shape[1]])
    
    if outputmap['tt'].ndim ==1:
        for i in range(outputmap['Qxy'].shape[1]):
            outputmap['Qz'][:,i] = k_i * (np.sin(np.radians(outputmap['tt'])) + np.sin(np.radians(alpha_i)))
            outputmap['Qxy'][:,i] = k_i * np.sqrt(1+(np.cos(np.radians(outputmap['tt'])))**2 - 2*np.cos(np.radians(outputmap['tt']))*np.cos(np.radians(outputmap['tth'][0,i]))) * ((outputmap['tth'][0,i] >0)/0.5-1)
    else:
        outputmap['Qz'] = k_i * (np.sin(np.radians(outputmap['tt'])) + np.sin(np.radians(alpha_i)))
        outputmap['Qxy'] = k_i * np.sqrt(1+(np.cos(np.radians(outputmap['tt'])))**2 - 2*np.cos(np.radians(outputmap['tt']))*np.cos(np.radians(outputmap['tth']))) * ((outputmap['tth'] >0)/0.5-1)

    if absQxy:
        outputmap['Qxy'] = np.abs(outputmap['Qxy'])
        
    return outputmap

#%% rebin
def qxyrebin(inputmap, dqxy = 0.0025, **kwargs):
    """
    rebin qxy:
        with pixel splitting
    qxy limit is the closest value from the limit in the original matrix
    input: n*m intensity matrix and qxy matrix, qz-array
    argument: dqxy = 0.0025
    unit: A^-1
    output: n'*m' intensity matrix, m'-array for qxy, n'-array for qz
    note: nan value for the Qxy region near specular
    note2: nan value for the Qxy/Qz outside of the limit (limit either created or getting from the result of the Qzrebin)
    do it by getting Qz broder mask from Qzrebin and multiply it to the mask, then temporarily set nan intensity to -1, in the end set everything outside the Qxy border to nan
    """
    # dq = 0.01A^-1 at 15keV, tth=11deg from Soller (dtheta = 0.04deg)
    # scan step: 1/4 of dq
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
    
    #Qz border mask from the qz-rebin: outside the border weight is zero
    if inputmap['QzborderMask']!=[]:
        mask = mask*inputmap['QzborderMask']
        print('apply Qz border mask')
    
    # start
    outputmap = copy.deepcopy(inputmap)
    
    # getting Qxy limits of the input
    outputmap['qxylim_raw'] = np.zeros((inputmap['mat'].shape[0],2))
    for idx in range(outputmap['qxylim_raw'].shape[0]):
        finite_idx = np.where(np.isfinite(inputmap['mat'][idx,:]))
        outputmap['qxylim_raw'][idx,0] = np.min(inputmap['Qxy'][idx, finite_idx[0]])
        outputmap['qxylim_raw'][idx,1] = np.max(inputmap['Qxy'][idx, finite_idx[0]])
    
    # replace the nan into -1, their weight will be 0
    np.place(inputmap['mat'],np.isnan(inputmap['mat']), -1.0)
    np.place(outputmap['mat'],np.isnan(outputmap['mat']), 0.0)
                
    # GISAXS specular preparation: getting Qxy boundary around the specular
    Qxy_boundary = np.zeros((inputmap['mat'].shape[0],2))
    for idx in range(Qxy_boundary.shape[0]):
        pos_idx = np.where(inputmap['Qxy'][idx,:]>0)
        Qxy_boundary[idx,0] = np.min(inputmap['Qxy'][idx, pos_idx[0]])
        neg_idx = np.where(inputmap['Qxy'][idx,:]<0)
        if neg_idx[0].size != 0:
            Qxy_boundary[idx,1] = np.max(inputmap['Qxy'][idx, neg_idx[0]])
        else:
            Qxy_boundary[idx,1] = Qxy_boundary[idx,0] 
    
    # create rebinned Qxy array from the lower and the upper lim of the initial qxy
    qxylim = [np.floor(np.min(inputmap['Qxy'])/dqxy)*dqxy, np.ceil(np.max(inputmap['Qxy'])/dqxy)*dqxy]
    outputmap['Qxy'] = np.array([np.arange(qxylim[0]-2*dqxy, qxylim[1]+dqxy*2, dqxy)])
    # prepare for pixel splitting: idx, stat and the rebinned intensity matrix
    # idx :,:,0 is the index of the upper edge of the Qxy bin where each pixel belongs to; 
    # idx :,:,1 the fraction of the pixel into the upper edge of the Qxy bin, the rest goes to the lower edge
    outputmap['idx'] = np.zeros([inputmap['mat'].shape[0],inputmap['mat'].shape[1],2])
    # stat: statistics matrix for the rebinning: accumulated fraction into each rebined pixel, to be used for normalization, all values set to zero
    outputmap['stat'] = np.zeros([inputmap['mat'].shape[0],outputmap['Qxy'].shape[1]])
    # result matrix for intensity, all values set to 0
    outputmap['mat'] = np.zeros([inputmap['mat'].shape[0],outputmap['Qxy'].shape[1]])
    
    # pixel splitting:
    #   1. use np.digitize find the upper edge index in the rebinned matrix for each original pixel
    #   2. for each original pixel, compute its fraction for the upper edge index
    #   3. split the fraction of each pixel into the upper and the lower edge in the rebinning statistics matrix; the values in the rebinning statistics matrix starts accumulation
    #   4. split the intensity of each pixel into the upper and the lower edge in the rebinned matrix; the intensity in the rebinned matrix starts accumulation
    # prepare function
    create_digitize = np.digitize
    for i in range(inputmap['Qxy'].shape[0]):
        outputmap['idx'][i,:,0] = create_digitize(inputmap['Qxy'][i,:], outputmap['Qxy'][0,:], right = True)
        #print(outputmap['idx'][i,:,0])
        for j in range(inputmap['Qxy'].shape[1]):
            idx_bins = int(outputmap['idx'][i,j,0])
            #print([i, j, idx_bins])
            outputmap['idx'][i,j,1] = 1-(outputmap['Qxy'][0,idx_bins]-inputmap['Qxy'][i,j])/dqxy
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
        # set mat element within Qxy boundary to nan for GISAXS
        np.place(outputmap['mat'][i,:],(outputmap['Qxy'] - Qxy_boundary[i,0])*(outputmap['Qxy'] - Qxy_boundary[i,1])<0, np.nan)
        # set mat element beyond Qxy limit to nan
        np.place(outputmap['mat'][i,:],(outputmap['Qxy'] - outputmap['qxylim_raw'][i,0])*(outputmap['Qxy'] - outputmap['qxylim_raw'][i,1])>0, np.nan)
    
    del empty_cell, miss_values, filled_cell, filled_value, i, j 
    return outputmap

def qzrebin(inputmap, dqz = 0.0015, **kwargs):
    """
    rebin qz, when qz is a n*m-matrix:
        with pixel splitting
    qz limit is the closest value from the limit in the original matrix
    input: n*m matrix for intensity, qxy, qz
    argument: dqz = 0.0015
    unit: A^-1
    output: n*m matrix for intensity and qxy, n-array for qz
    works with the same mechanism as the qxy rebin
    also output the Qz limit into the output field and set the int value outside the Qz border to nan for each column
    output also a Qz border mask for the next step
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
    
    # getting Qz limits of the original input
    outputmap['qzlim_raw'] = np.zeros((2,inputmap['mat'].shape[1]))
    for idx in range(outputmap['qzlim_raw'].shape[1]):
        unmask_idx = np.where(mask[:,idx]>0.5)
        if unmask_idx[0]!=[]:
            outputmap['qzlim_raw'][0,idx] = np.max(inputmap['Qz'][unmask_idx[0],idx])
            outputmap['qzlim_raw'][1,idx] = np.min(inputmap['Qz'][unmask_idx[0],idx])
        else:
            outputmap['qzlim_raw'][0,idx] = np.nan
            outputmap['qzlim_raw'][1,idx] = np.nan
    
    qzlim = [np.floor(np.min(inputmap['Qz'])/dqz)*dqz, np.ceil(np.max(inputmap['Qz'])/dqz)*dqz]
    #print('qzlim:\n%f\n%f\n' %(qzlim[0], qzlim[1]))
    outputmap['Qz'] = np.arange(qzlim[0]-2*dqz, qzlim[1]+2*dqz, dqz)
    #print(len(outputmap['Qz']))
    # prepare for pixel splitting: idx, stat and the rebinned intensity matrix
    # idx :,:,0 is the index of the upper edge of the Qz bin where each pixel belongs to; 
    # idx :,:,1 the fraction of the pixel into the upper edge of the Qz bin, the rest goes to the lower edge
    outputmap['idx'] = np.zeros([inputmap['mat'].shape[0],inputmap['mat'].shape[1],2])
    # stat: statistics matrix for the rebinning: accumulated fraction into each rebined pixel, to be used for normalization, all values set to zero
    outputmap['stat'] = np.zeros([outputmap['Qz'].shape[0],inputmap['mat'].shape[1]])
    Qxy_stat = np.zeros([outputmap['Qz'].shape[0],inputmap['mat'].shape[1]])
    # result matrix for intensity, all values set to 0
    outputmap['mat'] = np.zeros([outputmap['Qz'].shape[0],inputmap['mat'].shape[1]])
    outputmap['Qxy'] = np.zeros([outputmap['Qz'].shape[0],inputmap['mat'].shape[1]])    
    # pixel splitting:
    #   1. use np.digitize find the upper edge index in the rebinned matrix for each original pixel
    #   2. for each original pixel, compute its fraction for the upper edge index
    #   3. split the fraction of each pixel into the upper and the lower edge in the rebinning statistics matrix; the values in the rebinning statistics matrix starts accumulation
    #   4. split the intensity of each pixel into the upper and the lower edge in the rebinned matrix; the intensity in the rebinned matrix starts accumulation
    # prepare function
    create_digitize = np.digitize
    for i in range(inputmap['Qz'].shape[1]):
        outputmap['idx'][:,i,0] = create_digitize(inputmap['Qz'][:,i], outputmap['Qz'], right = True)
        #print(outputmap['idx'][i,:,0])
        for j in range(inputmap['Qz'].shape[0]):
            idx_bins = int(outputmap['idx'][j,i,0])
            #print([j, i, idx_bins])
            outputmap['idx'][j,i,1] = 1-(outputmap['Qz'][idx_bins]-inputmap['Qz'][j,i])/dqz
            outputmap['stat'][idx_bins,i] = outputmap['stat'][idx_bins,i] + outputmap['idx'][j,i,1]*mask[j,i]
            outputmap['stat'][idx_bins-1,i] = outputmap['stat'][idx_bins-1,i] + (1-outputmap['idx'][j,i,1])*mask[j,i]
            outputmap['mat'][idx_bins,i] = outputmap['mat'][idx_bins,i] + inputmap['mat'][j,i]*outputmap['idx'][j,i,1]*mask[j,i]
            outputmap['mat'][idx_bins-1,i] = outputmap['mat'][idx_bins-1,i] + inputmap['mat'][j,i]*(1-outputmap['idx'][j,i,1])*mask[j,i]
            Qxy_stat[idx_bins,i] = Qxy_stat[idx_bins,i] + outputmap['idx'][j,i,1]
            Qxy_stat[idx_bins-1,i] = Qxy_stat[idx_bins-1,i] + (1-outputmap['idx'][j,i,1])          
            outputmap['Qxy'][idx_bins,i] = outputmap['Qxy'][idx_bins,i] + inputmap['Qxy'][j,i]*outputmap['idx'][j,i,1]
            outputmap['Qxy'][idx_bins-1,i] = outputmap['Qxy'][idx_bins-1,i] + inputmap['Qxy'][j,i]*(1-outputmap['idx'][j,i,1])
            #print([i, outputmap['stat'][i,213]])
    
    # prepare function
    create_argwhere = np.argwhere
    # normalization by the statistics, everything with no pixel in is counted as negative (-1e6)
    np.place(outputmap['stat'], outputmap['stat']==0, -1)    
    np.place(Qxy_stat, Qxy_stat==0, -1)  
    empty_cell = create_argwhere(outputmap['stat']<0)
    for y,x in empty_cell:
        outputmap['mat'][y,x] = 1e6
    outputmap['mat'] = outputmap['mat'] / outputmap['stat']    
    outputmap['Qxy'] = outputmap['Qxy'] / Qxy_stat
    #test = copy.deepcopy(outputmap['mat'])
    
    # remove the columns without the values
    empty_col = []
    empty_row_limit = outputmap['mat'].shape[0]-2 
    for idx_value in range(outputmap['mat'].shape[1]):
        empty_cell = create_argwhere(outputmap['mat'][:,idx_value]<-1e3)[:,0]
        if len(empty_cell)>empty_row_limit:
            empty_col.append(idx_value)
    
    outputmap['mat'] = np.delete(outputmap['mat'], empty_col, axis = 1)
    outputmap['Qxy'] = np.delete(outputmap['Qxy'], empty_col, axis = 1)
    outputmap['qzlim_raw'] = np.delete(outputmap['qzlim_raw'], empty_col, axis = 1)
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
    outputmap['Qxy'] = np.delete(outputmap['Qxy'], range(0, idx_value+2), axis = 0)
    outputmap['Qz'] = np.delete(outputmap['Qz'], range(0, idx_value+2), axis = 0)
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
    outputmap['Qxy'] = np.delete(outputmap['Qxy'], range(idx_last-idx_value-1, idx_last+1), axis = 0)
    outputmap['Qz'] = np.delete(outputmap['Qz'], range(idx_last-idx_value-1, idx_last+1), axis = 0)
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
        filled_Qxy = create_zeros(len(filled_cell))
        for j in range(len(filled_cell)):
            filled_value[j] = outputmap['mat'][filled_cell[j],i]
            filled_Qxy[j] = outputmap['Qxy'][filled_cell[j],i]
#        print(i)
#        print(empty_cell)
#        print(filled_cell)
#        print(filled_value)
#        if filled_cell.size:
#            miss_values = np.interp(empty_cell, filled_cell, filled_value)
#            miss_Qxy = np.interp(empty_cell, filled_cell, filled_Qxy)
#        else:
#            miss_values = np.zeros(len(empty_cell))
#            miss_Qxy = np.zeros(len(empty_cell))
#            for j in range(len(empty_cell)):
#                miss_values[j] = outputmap['mat'][empty_cell[j], i-1]
#                miss_Qxy[j]=outputmap['Qxy'][empty_cell[j], i-1]
        miss_values = create_interp(empty_cell, filled_cell, filled_value)
        #print(miss_values)
        miss_Qxy = create_interp(empty_cell, filled_cell, filled_Qxy)                
        for j in range(len(empty_cell)):
            outputmap['mat'][empty_cell[j],i]=miss_values[j]
            outputmap['Qxy'][empty_cell[j],i]=miss_Qxy[j]
        
        # set int value outside of Qz limit raw to nan for every Qxy column
        if np.isnan(outputmap['qzlim_raw'][0,i]):
            outputmap['mat'][:,i] = np.nan
        else:
            np.place(outputmap['mat'][:,i],(outputmap['Qz'] - outputmap['qzlim_raw'][0,i])*(outputmap['Qz'] - outputmap['qzlim_raw'][1,i])>0, np.nan)
    
    # create a border mask: outside of the Qz border the weight is zero
    outputmap['QzborderMask'] = np.ones([outputmap['mat'].shape[0],outputmap['mat'].shape[1]])
    nan_cell = create_argwhere(np.isnan(outputmap['mat']))
    for y,x in nan_cell:
        outputmap['QzborderMask'][y,x] = 0
    
    del empty_cell, miss_values, filled_cell, filled_value, miss_Qxy, filled_Qxy, nan_cell, i, j 
    return outputmap

#%% general reading and export
def read_2D(filename, path, axis = ['Qxy', 'Qz']):
    """
    read the processed I(Qxy, Qz) or I(tth, tt) into a 2D matrix
    name as str
    path as str, end with //
    """

    # create dictionary for the data
    outputmap = {axis[0]: [], axis[1]: [], 'mat': []}
    
    # load files    
    file_intensity = path+filename+"_I.dat"
    file_xaxis = path+filename+"_"+axis[0]+".dat"
    file_yaxis = path+filename+"_"+axis[1]+".dat"
    outputmap['mat'] = np.loadtxt(file_intensity)
    outputmap[axis[0]] = np.loadtxt(file_xaxis, ndmin = 2)
        
    axis_1 =  np.loadtxt(file_yaxis)
    if outputmap[axis[0]].shape[0]>1 and axis_1.ndim==1:
        outputmap[axis[1]] = np.zeros([outputmap['mat'].shape[0],outputmap['mat'].shape[1]])
        for i in range(outputmap[axis[1]].shape[1]):
            outputmap[axis[1]][:,i] = axis_1
    else:
        outputmap[axis[1]]=axis_1
            
    del axis_1
    return outputmap

def export(inputmap, filename, exp_path, axis = ['Qxy', 'Qz']):
    """
    export the map into three ascii files: _I.dat, _[axis0].dat, _[axis1].dat
    """
    np.savetxt(exp_path+filename+"_I.dat",inputmap['mat'],fmt='%f')
    np.savetxt(exp_path+filename+"_"+axis[0]+".dat",inputmap[axis[0]],fmt='%f')
    np.savetxt(exp_path+filename+"_"+axis[1]+".dat",inputmap[axis[1]],fmt='%f')
