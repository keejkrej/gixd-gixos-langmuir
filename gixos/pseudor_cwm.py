#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PseudoR CWM (Capillary Wave Model) calculations
Translated from MATLAB code
@author: Chen
"""

import numpy as np
from scipy import constants
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from .gixos_calc import GIXOSCalculator

class PseudoRCWM:
    def __init__(self, config=None):
        """
        Initialize PseudoR CWM calculator
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary with the following keys:
            - energy: X-ray energy in eV
            - alpha_i: Incident angle in degrees
            - Ddet: Sample-detector distance in mm
            - footprint: Footprint size in mm
            - Qc: Critical Qz value
            - qxy0: Array of qxy0 values
            - qxy_bkg: Background qxy range
            - RqxyHW: Resolution for XRR
            - DSresHW: HW of DS resolution at specular
            - tension: Surface tension in N/m
            - temp: Temperature in K
            - amin: Minimal wavelength cutoff
        """
        # Default configuration
        self.config = {
            'geometry': {
                'Qc': 0.0216,
                'energy': 15000,
                'alpha_i': 0.07,
                'Ddet': 560.7,
                'pixel': 0.075,
                'footprint': 30,
                'qxy0': np.arange(0.01, 0.13, 0.01),
                'qxy_bkg': np.arange(0.35, 0.46, 0.01),
                'RqxyHW': 0.002,
                'DSresHW': 0.003,
                'qz_selected': np.array([0.1, 0.2, 0.35, 0.4, 0.45, 0.6])
            },
            'physics': {
                'kb': constants.Boltzmann,  # Boltzmann constant in J/K
                'tension': 0.073,  # Surface tension in N/m
                'temp': 298,  # Temperature in K
                'amin': 1e-10  # Minimal wavelength cutoff in m
            }
        }
        
        if config:
            self.config.update(config)
            
        # Initialize GIXOS calculator
        self.gixos = GIXOSCalculator(
            energy=self.config['geometry']['energy'],
            alpha_i=self.config['geometry']['alpha_i'],
            Ddet=self.config['geometry']['Ddet'],
            footprint=self.config['geometry']['footprint'],
            Qc=self.config['geometry']['Qc']
        )
        
        # Calculate derived parameters
        self._calculate_derived_parameters()

    def _calculate_derived_parameters(self):
        """Calculate derived parameters"""
        geo = self.config['geometry']
        phys = self.config['physics']
        
        # Wavelength and wave number
        geo['wavelength'] = 12404 / geo['energy']
        geo['wave_number'] = 2 * np.pi / geo['wavelength']
        
        # DS resolution
        geo['DSqxyHW'] = 2 * geo['DSresHW']
        
        # Maximum Q
        phys['Qmax'] = np.pi / phys['amin']
        
        # kbT/gamma prefactor
        phys['kbT_gamma'] = phys['kb'] * phys['temp'] / phys['tension'] * 1e20

    def calculate_ds_rrf_integ(self, beta_space, qxy0):
        """
        Calculate DS/RRF ratio by double integration over beta and phi
        
        Parameters:
        -----------
        beta_space : array_like
            Beta values in degrees
        qxy0 : float
            Qxy at beta = 0 degrees
            
        Returns:
        --------
        tuple
            (DS_RRF, DS_term, RRF_term)
        """
        geo = self.config['geometry']
        phys = self.config['physics']
        
        # Calculate Qz space
        qz_space = (np.sin(np.radians(geo['alpha_i'])) + 
                   np.sin(np.radians(beta_space))) * geo['wave_number']
        
        # Calculate phi range
        phi = 2 * np.degrees(np.arcsin(qxy0 / (2 * geo['wave_number'])))
        phi_HWHM = np.degrees(geo['DSqxyHW'] * geo['wavelength'] / (2 * np.pi))
        phi_upper = phi + phi_HWHM
        phi_lower = phi - phi_HWHM
        
        # Calculate beta range
        beta_upper = beta_space + geo['DSresHW']
        beta_lower = beta_space - geo['DSresHW']
        
        # Calculate tension terms
        eta = phys['kbT_gamma'] / (2 * np.pi) * qz_space**2
        xi = (2**(1-eta) * np.exp(np.log(np.abs(np.gamma(1-0.5*eta))) - 
                                 np.log(np.abs(np.gamma(0.5*eta)))) * 
             2 * np.pi / qz_space**2)
        
        # Calculate RRF term
        RRF_term = (geo['RqxyHW'] / phys['Qmax'])**eta
        
        # Calculate DS term via integral
        DS_term = np.ones(len(beta_space))
        for idx in range(len(beta_space)):
            def integrand(tt, tth):
                return self._integral_delta_beta_delta_phi(
                    tt, tth, phys['kbT_gamma'], 
                    geo['wave_number'], geo['alpha_i'], 
                    phys['Qmax']
                )
            
            DS_term[idx] = dblquad(
                integrand,
                phi_lower, phi_upper,
                lambda x: beta_lower[idx],
                lambda x: beta_upper[idx]
            )[0]
        
        DS_RRF = DS_term / RRF_term
        
        return DS_RRF, DS_term, RRF_term

    def _integral_delta_beta_delta_phi(self, beta, phi, kbT_gamma, wave_number, alpha, Qmax):
        """
        Integration over beta resolution and phi resolution
        
        Parameters:
        -----------
        beta : float
            Beta angle in degrees
        phi : float
            Phi angle in degrees
        kbT_gamma : float
            kbT/gamma prefactor
        wave_number : float
            Wave number
        alpha : float
            Alpha angle in degrees
        Qmax : float
            Maximum Q value
            
        Returns:
        --------
        float
            Integration result
        """
        # Calculate qxy and qz
        qxy = wave_number * np.sqrt(
            (np.cos(np.radians(beta)) * np.sin(np.radians(phi)))**2 +
            (np.cos(np.radians(alpha)) - np.cos(np.radians(beta)) * np.cos(np.radians(phi)))**2
        )
        qz = wave_number * (np.sin(np.radians(alpha)) + np.sin(np.radians(beta)))
        
        # Calculate eta and result
        eta = kbT_gamma / (2 * np.pi) * qz**2
        result = kbT_gamma * qxy**(eta-2) / Qmax**eta
        
        return result

    def plot_results(self, qz_space, DS_RRF, qxy0):
        """
        Plot DS/RRF results
        
        Parameters:
        -----------
        qz_space : array_like
            Qz values
        DS_RRF : array_like
            DS/RRF values
        qxy0 : float
            Qxy0 value
        """
        plt.figure('GIXOS factor')
        plt.plot(qz_space, DS_RRF/DS_RRF[0], '-', linewidth=1.5,
                label=f'Qxy,0={qxy0:.3f} Å⁻¹')
        plt.xlabel('Qz (Å⁻¹)')
        plt.ylabel('DS/RRF (normalized)')
        plt.legend()
        plt.grid(True) 