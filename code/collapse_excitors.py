#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for classes `Excitor` and `Cluster`.

@author: mz407
"""
import numpy as np


class Excitor():
    """
    Excitor object, containing integer representations of bit strings for the
    the excitor, its creation and annihilation and its amplitude if known.
    """
    def __init__(self, excitor, amplitude, ref=0, err=0):
        """
        Parameters:
            excitor int: integer representation of bit string of the excitor.
            amplitude float: excitor amplitude.
            ref int: integer representation of bit string of the
                     reference determinant.
        """
        self.excitor = excitor
        self.amplitude = amplitude
        self.excitation = 0
        self.annihilation = 0
        self.creation = 0
        self.err = err
        if ref:
            self.decompose(ref)

    def decompose(self, ref):
        """
        Decompose the excitor into annihilation and creation w.r.t. a reference
        """
        self.excitation = self.excitor ^ ref
        self.annihilation = self.excitation & ref
        self.creation = self.excitation & self.excitor
        self.exlevel = bin(self.annihilation).count("1")

    def __repr__(self):
        return "<Configuration: {}, Amplitude: {}, Error: {}>".format(self.excitor,
                                                           self.amplitude, 
                                                           self.err)

    def __str__(self):
        return str(self.excitor)


class Cluster(Excitor):
    """
    Cluster object.
    """
    def __init__(self, ref, amplitude=1):
        """
        Parameters:
            ref int: integer representation of bit string of the reference.
            amplitude float: excitor amplitude.
        """
        self._ref = ref
        self.history = ''
        Excitor.__init__(self, ref, amplitude)
    
    def collapse_excitor(self, excitor1):
        """
        Combining the cluster with an extra excitor. The resultant cluster
        operator is changed in place.

        Returns:
            False if invalid combination.
            1 / -1 sign of resultant combination.
        """
        # Check if valid
        if (self.annihilation & excitor1.annihilation) or\
           (self.creation & excitor1.creation):
            # Not Valid
            return False
        # Valid - consider sign change
        max_position = max(len(bin(self.excitor)), len(bin(excitor1.excitor)))-2
        sign = 1
        for position in range(max_position):
            if excitor1.annihilation & (1 << position):  # is annihilation
                # Find the number of created orbitals with lower index
                perm = bin(((1 << position)-1) & self.excitor).count("1")
                self.excitor -= (1 << position)
            elif excitor1.creation & (1 << position):  # is creation
                # Find all excitation operators with higher index
                perm = bin(excitor1.excitor >> (position + 1)).count("1")
                # Find all existing electrons with lower index
                perm += bin(((1 << position)-1) & self.excitor).count("1")
                self.excitor += (1 << position)
            else:  # neither creation nor annihilation
                continue
            if perm % 2 == 1:
                sign *= -1
        self.amplitude *= excitor1.amplitude * sign
        self.history += str(excitor1.exlevel)
        if excitor1.err:
            self.combine_error(excitor1)
        self.decompose(self._ref)
        return self

    def combine_error(self, excitor1):
        new_frac_err = np.linalg.norm([self.err/self.amplitude,
                                       excitor1.err/excitor1.amplitude])
        self.err = new_frac_err * self.amplitude
