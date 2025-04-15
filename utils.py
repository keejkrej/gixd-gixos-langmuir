import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def get_data_path(df, root_path):
    if 'sample' in df.columns and 'scan_id' in df.columns:
        if len(df['sample'].unique()) == 1:
            name = df['sample'].values[0]
            scan_id_list = df['scan_id'].values
            data_path = [root_path / str(name) / str(scan_id) for scan_id in scan_id_list]
            return data_path
    else:
        raise ValueError('Multiple samples in metadata')
    
def get_data(data_path):
    dat_files = list(data_path.glob('*.dat'))
    scan_id = str(data_path).split('/')[-1]
    
    if len(dat_files) != 3:
        raise ValueError('Missing data files')
    else:
        for dat_file in dat_files:
            if dat_file.name.endswith('_I.dat'):
                I = np.loadtxt(dat_file)
            elif dat_file.name.endswith('_Qxy.dat'):
                Qxy = np.loadtxt(dat_file)
            elif dat_file.name.endswith('_Qz.dat'):
                Qz = np.loadtxt(dat_file)
    
    return I, Qxy, Qz, scan_id

def get_crop(roi, I, Q_xy, Q_z):
    Q_xy_min, Q_xy_max, Q_z_min, Q_z_max = roi
    if Q_xy_min == None:
        Q_xy_min = Q_xy[0,0]
    if Q_xy_max == None:
        Q_xy_max = Q_xy[-1,-1]
    if Q_z_min == None:
        Q_z_min = Q_z[0,0]
    if Q_z_max == None:
        Q_z_max = Q_z[-1,-1]
    mask = (Q_xy > Q_xy_min) & (Q_xy < Q_xy_max) & (Q_z > Q_z_min) & (Q_z < Q_z_max)
    I_copy = I.copy()
    I_copy[~mask] = np.nan
    return I_copy

def get_avg(roi, I, Q_xy, Q_z):
    crop = get_crop(roi, I, Q_xy, Q_z)
    avg = np.nanmean(crop, axis=0)
    return avg

def low_pass_filter(signal, window_size=None):
    if window_size is None:
        return signal
    window = np.ones(window_size) / window_size
    filtered_signal = np.convolve(signal, window, mode='same')
    return filtered_signal

def lorentzian(Q_xy, I_max, Q_xy_max, w_xy, B, C):
    return I_max / (1 + ((Q_xy - Q_xy_max) / w_xy) ** 2) + B * Q_xy + C

def fit_lorentzian(Q_xy, I):
    popt, pcov = curve_fit(lorentzian, Q_xy, I, 
                        p0=[np.max(I), Q_xy[np.argmax(I)], 0.1, 0, 0])

    Q_xy_fit = np.linspace(Q_xy[0], Q_xy[-1], 1000)
    I_fit = lorentzian(Q_xy_fit, *popt)

    return popt, pcov, Q_xy_fit, I_fit

def replace_peak(I, Q_xy, Q_xy_range_list):
    for Q_xy_range in Q_xy_range_list:
        mask = (Q_xy[0] > Q_xy_range[0]) & (Q_xy[0] < Q_xy_range[1])
        I_crop = I[mask]
        I_start = I_crop[0]
        I_end = I_crop[-1]
        I[mask] = np.linspace(I_start, I_end, len(I_crop))
    return I

