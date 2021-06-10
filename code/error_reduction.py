#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-series visualizations of various ways
to reduce stochastic error of energy estimator.

@author: mz407
"""
import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from proj_energy_cc import separate_projected_e, read_fci_space_index


class ProjectionEstimator():
    """ProjectionEstimator"""
    def __init__(self, basedir, rsdir):
        self.basedir = basedir
        self.wfnfile = os.path.join(basedir, 'FCI_WFN')
        self.rsdir = os.path.join(basedir, rsdir)
        self.hamilfile = os.path.join(basedir, 'HAMIL')
        self.sep_proje_reduced = None
        self.ci_labels_reduced = None
        self.sep_proje_train = None
        self.sep_proje_eval = None
        self.proje_var = None
        self.proje_corr = None
        self.proje_cov = None
        self.result = None
        self.errors = None
        self.expcor_err = None
        self.means = None
        self.stderr = None

    def readfiles(self, readno, start, level, interval=1):
        """Read in files for the analysis."""
        filelist = [os.path.join(self.rsdir, f'HANDE.RS.{i}.p0.H5')
                    for i in range(start, start+readno, interval)]
        ci_labels = read_fci_space_index(self.wfnfile)
        sep_proje = separate_projected_e(self.hamilfile, self.wfnfile,
                                         filelist, level)
        finite_rows = np.isfinite(sep_proje).all(axis=1)
        self.sep_proje_reduced = sep_proje[finite_rows]
        self.ci_labels_reduced = ci_labels[finite_rows]

    def splittraineval(self, cut=None, eval_all=True):
        """Split data into something resembling train and test sets,
        to avoid the overfitting of covariance matrix."""
        if cut is None:
            cut = self.sep_proje_reduced.shape[1]
        self.sep_proje_train = self.sep_proje_reduced[:, :cut]
        if eval_all:
            self.sep_proje_eval = self.sep_proje_reduced[:, :]
        else:
            self.sep_proje_eval = self.sep_proje_reduced[:, cut:]
        self.proje_corr = np.corrcoef(self.sep_proje_train)
        self.proje_cov = np.cov(self.sep_proje_train)
        self.proje_var = np.diag(self.proje_cov)

    def _projections(self, nvar):
        """
        Sub-routine to calculate the indepent and correlated estimators
        of projected energies.
        """
        min_var = self.proje_var.argsort()[:nvar]
        add_coeffs = 1 / self.proje_var[min_var]
        indp_est_proje = np.dot(add_coeffs, self.sep_proje_eval[min_var]) /\
            np.sum(add_coeffs)

        # consider covariance
        coverr = []
        try:
            proje_cov_inv = np.linalg.inv(self.proje_cov[min_var][:, min_var])
            cov_weight = np.sum(proje_cov_inv, axis=0) / np.sum(proje_cov_inv)
            cov_est_proje = np.dot(cov_weight, self.sep_proje_eval[min_var])
            coverr.append(1/np.sum(proje_cov_inv))
        except:
            cov_est_proje = np.ones(self.sep_proje_eval.shape[1])
            cov_est_proje[:] = np.nan
            coverr.append(np.nan)
        return np.array([indp_est_proje, cov_est_proje])

    def _corrtheoerr(self, nvar):
        """
        To calculate the theoretical error of correlated estimator.
        """
        min_var = self.proje_var.argsort()[:nvar]
        proje_cov_inv = np.linalg.inv(self.proje_cov[min_var][:, min_var])
        return 1/np.sum(proje_cov_inv)

    def eval_combinations(self, max_dets=None):
        """
        The main process of evaluating the estimators of projected energies.
        """
        self.max_dets = max_dets
        self.result = np.array([self._projections(i) for i in range(1, self.max_dets)])
        self.errors = np.std(self.result, axis=2).T
        self.expcor_err = np.array([np.sqrt(self._corrtheoerr(i))
                                    for i in range(1, self.max_dets)]) / \
            np.sqrt(self.result.shape[2])
        self.means = np.average(self.result, axis=2).T
        self.stderr = self.errors / np.sqrt(self.result.shape[2])

    def plot_insta_err(self, ax=None):
        """
        Plot instantaneous errors of the estimators.
        """
        if ax is None:
            ax = plt.gca()
        ax.set_yscale('log')
        ax.plot(list(range(1, self.max_dets)), self.errors[0], label='independent')
        ax.plot(list(range(1, self.max_dets)), self.errors[1], label='correlated')
        ax.set_ylabel(r'Stochastic error in $E_\mathrm{corr}$ / ha')
        ax.set_xlabel('Number of determinants in estimator')
        ax.axhline(np.sqrt(self.proje_var[0]), linestyle='--', color='black', label='reference')
        ax.legend()
        return ax

    def plot_err(self, expect=False, axallerr=None):
        """
        Plot the standard errors of means of the estimators.
        """
        if axallerr is None:
            axallerr = plt.gca()
        axallerr.set_yscale('log')
        axallerr.plot(list(range(1, self.max_dets)), self.stderr[0], label='independent')
        axallerr.plot(list(range(1, self.max_dets)), self.stderr[1], label='correlated (actual)')
        if expect:
            axallerr.plot(list(range(1, self.max_dets)), self.expcor_err, label='correlated (expected)')
        axallerr.set_ylabel(r'Stochastic error in $\Delta E$ / ha')
        axallerr.set_xlabel('Number of determinants in estimator')
        axallerr.axhline(np.sqrt(self.proje_var[0]) / np.sqrt(self.result.shape[2]),
                         linestyle='--', color='black', label='reference')
        axallerr.legend()
        return axallerr

    def plot_mean(self, exact, exact_label='CCSD', axabs=None):
        """
        Plot expectation value of the estimators.
        """
        if axabs is None:
            axabs = plt.gca()
        axabs.plot(list(range(1, self.max_dets)), self.means[0], label='independent')
        axabs.plot(list(range(1, self.max_dets)), self.means[1], label='correlated')
        axabs.axhline(self.means[0, 0], linestyle='--', color='silver', label='reference')
        if exact:
            axabs.axhline(exact, linestyle='dashdot', color='green', label=exact_label)
        axabs.set_xlabel('Number of determinants in estimator')
        axabs.set_ylabel(r'$E_\mathrm{corr}$ / ha')
        axabs.legend()
        return axabs

    def plot_comparisons(self, exact, blocked, blockederr, axdelta=None):
        """
        Plot to compare the performance of the estimators with blocking
        analysis.
        """
        if axdelta is None:
            axdelta = plt.gca()
        delta = self.means - exact
        axdelta.errorbar(list(range(1, self.max_dets)), delta[0], yerr=self.stderr[0], label='independent')
        axdelta.errorbar(list(range(1, self.max_dets)), delta[1], yerr=self.stderr[1], label='correlated')
        axdelta.axhline(delta[0, 0], linestyle=':', color='grey', label='reference')
        axdelta.axhline(0, linestyle='-', linewidth=1, color='black')
        if blocked:
            axdelta.axhline(blocked-exact, linestyle='--', color='darkgreen', label='reblocked')
        if blockederr:
            axdelta.fill_between([0, self.max_dets], [blocked-exact-blockederr,blocked-exact-blockederr],
                    [blocked-exact+blockederr,blocked-exact+blockederr], color='green', alpha=0.2)
        axdelta.set_xlabel('Number of determinants in estimator')
        axdelta.set_ylabel(r'$E-E_\mathrm{CCSD}$ / ha')
        axdelta.legend()
        return axdelta

    def bootstrap(self, resamples, cut, max_dets, no_eval=0):
        """
        Perform bootstrap analysis to achieve a more realistic error estimate.
        """
        self.max_dets = max_dets
        rng = np.random.default_rng()
        bsresult = np.zeros((self.max_dets - 1, 2, resamples))
        readinproje = deepcopy(self.sep_proje_reduced)
        try:
            for resampleid in range(resamples):
                rng.shuffle(self.sep_proje_reduced, axis=1)
                if no_eval == 0:
                    self.splittraineval(cut, True)
                elif no_eval == -1:
                    self.splittraineval(cut, False)
                else:
                    self.splittraineval(cut, True)
                    self.sep_proje_eval = self.sep_proje_eval[:, rng.integers(0, self.sep_proje_eval.shape[1], no_eval)]
                self.eval_combinations(self.max_dets)
                bsresult[:, :, resampleid] = self.means[:].T
            # fill in all results
            self.result = bsresult
            self.errors = np.std(self.result, axis=2).T
            self.means = np.average(self.result, axis=2).T
            self.stderr = self.errors
        except np.linalg.LinAlgError:
            raise
        # reset
        finally:
            self.sep_proje_reduced = readinproje

    @property
    def max_dets(self):
        """max number of determinants considered in the plots."""
        return self._max_dets

    @max_dets.setter
    def max_dets(self, max_dets):
        if max_dets is None:
            max_dets = len(self.ci_labels_reduced)+1
        self._max_dets = max_dets
