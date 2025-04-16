# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:15:52 2023

@author: shenc
"""

#%% import
import p08_general.P08ScanTools as P08ScanTools
import p08_general.fio_reader as fio_reader
import p08_GIXD.p08_GIXD as GIXD
import GIXD_additional_dependency as GIXD_add
import p08_GIXD.GIXD_plot as plt
import matplotlib.pyplot as pyplt
import numpy as np
import copy
import platform, os
import glob

def gisaxs_process(scan_id, bkg_scan_id = False, data_folder = "../../raw/", processed_folder = "../../scratch_cc/", sdd = 560.7, poni = [990, 1020], orient = [-1, -1], rot3 = -0.6, det = "eiger1m", Yoneda_mode = "fix", anglerebin = [0.008, 0.008], qrebin = False, plot_option = False, colorscale = [1,5000]):
    '''
    anglerebin: 1st is on tt, 2nd is on tth
    '''
    try:
        os.mkdir(processed_folder)
    except FileExistsError:
        print("processed data desitination already exist")
        pass
    try:
        os.mkdir(processed_folder+'troughGISAXS/')
    except FileExistsError:
        print("troughGISAXS data desitination already exist")
        pass
    
    angularrebin_folder = processed_folder + "troughGISAXS/angularrebin/"
    try:
        os.mkdir(angularrebin_folder)
    except FileExistsError:
        print("angularrebin folder already exist")
        pass    
    
    if qrebin:
        qrebin_folder = processed_folder + "troughGISAXS/qrebin/"
        try:
            os.mkdir(qrebin_folder)
        except FileExistsError:
            print("qrebin folder already exist")
            pass

    # close all existing figures
    pyplt.close(fig='all') 
    # load data
    fio_addr_lst = glob.glob(data_folder + "*_"+"{:05d}".format(scan_id)+'.fio')
    fio_addr = fio_addr_lst[0]
    fio_addr = fio_addr.replace('\\', '/')
    sample = fio_addr[fio_addr.rindex('/raw/')+5:-10]
    fio_data = fio_reader.read(fio_addr)
    if 'eiger_roi2' not in fio_data[3]["signalcounter"]:
        print('scan %d is not a GISAXS scan\n' % scan_id)
        return
    else:
        det_label = "eiger"
        print('scan %d: load GISAXS image\n' % scan_id)    
    
    # counting time and integrated primary
    counting_time = np.sum(fio_data[2]['counting_time'])        
    integrated_primary = np.sum(fio_data[2]['cyber_cts'])  
    
    # load eiger data
    frame_data = P08ScanTools.Scan()
    frame_data.load_scan(fio_addr, auto_load_images = True)
    sample_img_array = np.sum(frame_data.image_data[det_label], axis = 0).T
    print("summing all frames up")
    del frame_data
    # create mask
    eiger1m_mask = np.where(sample_img_array<4e9, 1, 0)
    if bkg_scan_id:
        bkg_fio_addr_lst = glob.glob(data_folder + "*_"+"{:05d}".format(bkg_scan_id)+'.fio')
        bkg_fio_addr = bkg_fio_addr_lst[0]
        bkg_fio_addr = bkg_fio_addr.replace('\\', '/')
        bkg_fio_data = fio_reader.read(bkg_fio_addr)
        if 'eiger_roi2' in bkg_fio_data[3]["signalcounter"]:
            bkg_counting_time = np.sum(bkg_fio_data[2]['counting_time'])
            bkg_integrated_primary = np.sum(bkg_fio_data[2]['cyber_cts'])
            frame_data = P08ScanTools.Scan()
            frame_data.load_scan(bkg_fio_addr, auto_load_images = True)
            bkg_img_array = np.sum(frame_data.image_data[det_label], axis = 0).T
            del frame_data
            print("bkg data loaded and summed up and will be subtracted")
        else:
            bkg_counting_time = copy.deepcopy(counting_time)
            bkg_integrated_primary = copy.deepcopy(integrated_primary)
            bkg_img_array = np.zeros((sample_img_array.shape[0],sample_img_array.shape[1]))
            print("bkg scan is not an eiger scan")
    else:
        bkg_counting_time = copy.deepcopy(counting_time)
        bkg_integrated_primary = copy.deepcopy(integrated_primary)
        bkg_img_array = np.zeros((sample_img_array.shape[0],sample_img_array.shape[1]))
        print("no bkg")
    img_array = sample_img_array - bkg_img_array/bkg_integrated_primary*integrated_primary
    # load energy
    energy = copy.deepcopy(fio_data[0]['energyfmb'])
    # detector alpha_i and alpha_f angle
    alpha_i = np.abs(2.*fio_data[0]['om'])
    alpha_f = 0.0
    # prepare matrix: load lambda images into dictionary, convert pixel to q
    img_data = {'chh': [], 'chv': [],'mat': [], 'detrot': []}
    img_data['detrot'] = 0.0
    #pyplot.imshow(img_array)
    img_data['mat']= img_array  # vertical mounting
    #pyplot.imshow(img_data['mat'])
    img_data['chh'] = np.array([np.arange(0., img_data['mat'].shape[1])])
    img_data['chv'] = np.arange(0, img_data['mat'].shape[0])
    Itt=copy.deepcopy(GIXD.pixel2th(img_data, 0, sdd, poni[1], poni[0], Yoneda_mode, energy, det, orient, alpha_f, rot3))
    # rebin in tt
    Itt_vertrebin = GIXD_add.ttrebin(Itt, dtt = anglerebin[0], mask = eiger1m_mask)
    Itt_rebin = GIXD_add.tthrebin(Itt_vertrebin, dtth = anglerebin[1])
    # export
    filename = "GID_" + sample+"_"+"{:05d}".format(scan_id)+'_angle'
    GIXD.export(Itt_rebin, filename, angularrebin_folder, axis = ['tth', 'tt'])
    # plot
    if plot_option:
        fig_Itt_rebin = plt.plot2d(Itt_rebin, axis = ['tth', 'tt'], range_I = colorscale)
        plt.export(fig_Itt_rebin, filename, angularrebin_folder)
        del fig_Itt_rebin
    print('tt rebin finishes')
        
    if qrebin and len(qrebin) == 2:
        print('q rebin starts')
        Iq = GIXD.th2q(Itt, energy = energy, alpha_i = alpha_i, absQxy = False)
        Iq_qzrebin = GIXD.qzrebin(Iq, dqz=qrebin[0], mask = eiger1m_mask)
        Iq_rebin = GIXD.qxyrebin(Iq_qzrebin, dqxy = qrebin[1])
        filename_Iq = "GID_" + sample+"_"+"{:05d}".format(scan_id)
        GIXD.export(Iq_rebin, filename_Iq, qrebin_folder, axis = ['Qxy', 'Qz'])
        # plot
        if plot_option:
            fig_Iq_rebin = plt.plot2d(Iq_rebin, axis = ['Qxy', 'Qz'], range_I = colorscale)
            plt.export(fig_Iq_rebin, filename_Iq, qrebin_folder)
            del fig_Iq_rebin
        print('q rebin finishes')
        del Iq, Iq_qzrebin, Iq_rebin
    
    del sample_img_array, bkg_img_array, img_array, img_data, eiger1m_mask, Itt, Itt_vertrebin, Itt_rebin
    
def gisaxs_batch_process(beamtime_id, scan_id_range = [1, 99999], bkg_scan_id = False, sdd = 560.7, poni = [990, 1020], orient = [-1, -1], rot3 = -0.6, det = "eiger1m", Yoneda_mode = "fix", anglerebin = [0.008, 0.008], qrebin = False, plot_option = False, colorscale = [1,5000]):
    
    data_folder = '/asap3/petra3/gpfs/p08/2023/data/'+"{:d}".format(beamtime_id)+'/raw/'
    processed_folder = '/asap3/petra3/gpfs/p08/2023/data/'+"{:d}".format(beamtime_id)+'/processed/'
    
    if platform.uname()[0] == "Windows":
        data_folder = data_folder.replace("/asap3/petra3/gpfs/", "U:/")
        processed_folder = processed_folder.replace("/asap3/petra3/gpfs/", "U:/")
        print('windows system core')
    else:
        print('maxwell core')
        
    filename_lst = glob.glob(data_folder+'*.fio')
    scan_id_lst = []
    for filename in filename_lst:
        if int(filename[-9:-4])>=scan_id_range[0] and int(filename[-9:-4])<=scan_id_range[1]:
            scan_id_lst.append(int(filename[-9:-4]))
    print(scan_id_lst)
    
    for scan_id in scan_id_lst:
        gisaxs_process(scan_id, bkg_scan_id = bkg_scan_id, data_folder = data_folder, processed_folder = processed_folder, sdd = sdd, poni = poni, orient = orient, rot3 = rot3, det = det, Yoneda_mode = Yoneda_mode, anglerebin = anglerebin, qrebin = qrebin, plot_option = plot_option, colorscale = colorscale)
        
        