#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GIXOS calculation utilities
Translated from MATLAB code
@author: Chen
"""

import numpy as np
from scipy.integrate import quad

class GIXOSCalculator:
    def __init__(self, energy, alpha_i, Ddet, footprint, Qc):
        """
        Initialize GIXOS calculator with experimental parameters
        
        Parameters:
        -----------
        energy : float
            X-ray energy in eV
        alpha_i : float
            Incident angle in degrees
        Ddet : float
            Sample-detector distance in mm
        footprint : float
            Footprint size in mm
        Qc : float
            Critical Qz value
        """
        self.energy = energy
        self.alpha_i = alpha_i
        self.Ddet = Ddet
        self.footprint = footprint
        self.Qc = Qc
        
        # Constants
        self.planck = 1240.4  # eV*nm
        self.wavelength = self.planck / self.energy * 10  # Convert to Angstroms
        self.k = 2 * np.pi / self.wavelength
        
        # For Vineyard factor calculation
        self.beta = 1e-9  # For water
        self.alpha_c = np.degrees(np.arcsin(self.Qc / (2 * 2 * np.pi / (self.planck/self.energy * 10))))
        self.l_i = 1/np.sqrt(2) * np.sqrt(self.alpha_c**2 - np.radians(self.alpha_i)**2 + 
                                        np.sqrt((self.alpha_c**2 - np.radians(self.alpha_i)**2)**2 + 
                                               (2*self.beta)**2))
        self.normalization = (self.planck/self.energy * 10)/(2*np.pi)/self.l_i

    def fresnel(self, Qz):
        """
        Calculate Fresnel reflectivity
        
        Parameters:
        -----------
        Qz : array_like
            Qz values to calculate reflectivity for
            
        Returns:
        --------
        array_like
            Array containing [Qz, reflectivity] pairs
        """
        r = (Qz - np.sqrt(Qz**2 - self.Qc**2)) / (Qz + np.sqrt(Qz**2 - self.Qc**2))
        refl = r * np.conj(r)
        refl[refl > 1] = 1
        return np.column_stack((Qz, refl))

    def Tsqr(self, Qz):
        """
        Calculate transmission with footprint average
        
        Parameters:
        -----------
        Qz : array_like
            Qz values to calculate transmission for
            
        Returns:
        --------
        array_like
            Array containing [Qz, alpha_f, x, averaged_vf] values
        """
        result = np.zeros((len(Qz), 4))
        result[:, 0] = Qz
        
        # Calculate alpha_f
        result[:, 1] = np.degrees(np.arcsin(Qz / (2 * np.pi) * (self.planck/self.energy * 10) - 
                                          np.sin(np.radians(self.alpha_i))))
        
        # Calculate x
        result[:, 2] = result[:, 1] / np.degrees(np.arcsin(self.Qc / (2 * 2 * np.pi / (self.planck/self.energy * 10))))
        
        # Calculate averaged Vineyard factor
        for i in range(len(Qz)):
            result[i, 3] = self.ave_vf(result[i, 1])
            
        return result

    def dQz(self, Qz):
        """
        Calculate dQz from long footprint
        
        Parameters:
        -----------
        Qz : array_like
            Qz values to calculate dQz for
            
        Returns:
        --------
        array_like
            Array containing [Qz, alpha_f_center, alpha_f_max, alpha_f_min, dQz, dQz/Qz] values
        """
        result = np.zeros((len(Qz), 6))
        result[:, 0] = Qz
        
        # Calculate alpha_f center
        result[:, 1] = np.degrees(np.arcsin(Qz / (2 * np.pi) * (self.planck/self.energy * 10) - 
                                          np.sin(np.radians(self.alpha_i))))
        
        # Calculate alpha_f max and min
        result[:, 2] = np.degrees(np.arctan(np.tan(np.radians(result[:, 1])) * 
                                          self.Ddet / (self.Ddet - self.footprint)))
        result[:, 3] = np.degrees(np.arctan(np.tan(np.radians(result[:, 1])) * 
                                          self.Ddet / (self.Ddet + self.footprint)))
        
        # Calculate dQz
        result[:, 4] = 0.5 * ((np.sin(np.radians(result[:, 2])) + np.sin(np.radians(self.alpha_i))) * 
                             (2 * np.pi / (self.planck/self.energy * 10)) - 
                             (np.sin(np.radians(result[:, 3])) + np.sin(np.radians(self.alpha_i))) * 
                             (2 * np.pi / (self.planck/self.energy * 10)))
        
        # Calculate dQz/Qz
        result[:, 5] = result[:, 4] / Qz
        
        return result

    def ave_vf(self, alpha_fc):
        """
        Calculate averaged Vineyard factor along the footprint
        
        Parameters:
        -----------
        alpha_fc : float
            Center alpha_f value in degrees
            
        Returns:
        --------
        float
            Averaged Vineyard factor
        """
        step = int(np.floor(self.footprint / 5))
        positions = np.linspace(-5 * step/2, 5 * step/2, step + 1)
        temp = np.zeros(step + 1)
        
        for i, pos in enumerate(positions):
            temp[i] = self.vf_length_corr(alpha_fc, pos)
            
        return np.sum(temp) / (step + 1)

    def vf_length_corr(self, alpha_fc, length):
        """
        Calculate Vineyard factor with length correction
        
        Parameters:
        -----------
        alpha_fc : float
            Center alpha_f value in degrees
        length : float
            Position along the footprint
            
        Returns:
        --------
        float
            Vineyard factor with length correction
        """
        alpha_f = np.degrees(np.arctan(self.Ddet * np.tan(np.radians(alpha_fc)) / (self.Ddet - length)))
        return self.vineyard_factor(alpha_f)

    def vineyard_factor(self, alpha_f):
        """
        Calculate Vineyard factor
        = Transmission^2 * penetration_depth
        Penetration depth takes into account both in and out paths
        See Dosch's paper Phys Rev B (1987)
        
        Parameters:
        -----------
        alpha_f : float
            Exit angle in degrees
            
        Returns:
        --------
        float
            Vineyard factor
        """
        x = alpha_f / self.alpha_c
        
        if x > 0:
            # Calculate transmission
            T = np.abs(2 * x / (x + np.sqrt(x**2 - 1 - 2*self.beta*1j/self.alpha_c**2)))**2
            
            # Calculate penetration depth
            l_f = 1/np.sqrt(2) * np.sqrt(self.alpha_c**2 - np.radians(alpha_f)**2 + 
                                       np.sqrt((self.alpha_c**2 - np.radians(alpha_f)**2)**2 + 
                                              (2*self.beta)**2))
        else:
            T = 0
            l_f = 1/np.sqrt(2) * np.sqrt(self.alpha_c**2 - np.radians(alpha_f)**2 + 
                                       np.sqrt((self.alpha_c**2 - np.radians(alpha_f)**2)**2 + 
                                              (2*self.beta)**2))
        
        # Calculate Vineyard factor
        vf = (self.planck/self.energy * 10)/(2*np.pi) * T / (l_f + self.l_i) / self.normalization
        
        return vf 