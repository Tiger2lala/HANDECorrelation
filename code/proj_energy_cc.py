#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to read HANDE output and calculate projected energies.
"""
from copy import deepcopy
import h5py
import numpy as np
from collapse_excitors import Excitor, Cluster

"""
Need a dict of {'det#': coefficient}

Generate a list of Excitor objects.
Generate ref Cluster.

for no_of_product in product_limit:
    gen_list_of_clusters(ref, list_of_excitors, no_of_products)
    add the coefficients to the det list.

"""

"""
Generate list of clusters (ref, list_of_excitors, no_of_products)
if no_of_products > 2:
    list_cluster = generate_list_of_clusters(list_of_excitors, no_of_prod-1)
else:
    list_cluster = [ref]
new_list_cluster = [deepcopy(cluster).collapse_excitor(excitor)
                    for cluster in list_cluster for excitor in list_excitor
                    if deepcopy(cluster).collapse_excitor(excitor)]
return new_list_cluster
"""


def read_excitor(filename, refpop=None):
    """
    Read instantaneous population of each determinant from HANDE restart file.

    Returns:
        list of Excitor objects.
    """
    h5file = h5py.File(filename, 'r')
    det = h5file['qmc']['psips']['determinants'][:, 0]
    pop = h5file['qmc']['psips']['populations'][:, 0]
    if refpop is None:
        amp = pop / pop[0]  # get coefficient
    else:
        amp = pop / refpop
    excitors = [Excitor(exc, amp[idx]) for idx, exc in enumerate(det)]
    for excitor in excitors:
        excitor.decompose(excitors[0].excitor)
    return excitors


def read_hmat(filename):
    """
    Read and form Hamiltonian matrix from HANDE HAMIL file.
    """
    hmat_elem = np.loadtxt(filename)
    ndet = int(np.max(hmat_elem[:, 0]))
    hmat = np.zeros((ndet, ndet))
    for (row, col, entry) in hmat_elem:
        hmat[int(row)-1, int(col)-1] = entry
        hmat[int(col)-1, int(row)-1] = entry
    return hmat


def read_fci_space(filename):
    """
    Read determinant indices for FCI from FCI_WFN file.

    Returns:
        {determinant integer: index in hmat}
    """
    detlist = np.loadtxt(filename)[:, 0:2]
    detlist = detlist.astype(np.int32)
    return dict(zip(detlist[:, 1], detlist[:, 0]-1))


def read_fci_space_index(filename):
    """
    Read determinant indices for FCI from FCI_WFN file.

    Returns:
        np.array([determinant integers])
    """
    detlist = np.loadtxt(filename)[:, 1]
    detlist = detlist.astype(np.int32)
    return detlist


def _gen_cluster_product(clusters, excitors):
    new_cluster_list = [deepcopy(cluster).collapse_excitor(excitor)
                        for cluster in clusters for excitor in excitors
                        if deepcopy(cluster).collapse_excitor(excitor)]
    return new_cluster_list


def gen_cluster_operators(excitors, result_level):
    """
    Generate a list of list of all valid excitation operators, grouped
    by the excitation level.
    """
    # For now, assume the 1st element in excitors list is the reference.
    ref = Cluster(excitors[0].excitor)
    excitors = deepcopy(excitors[1:])
    all_operators = [[ref]]
    curr_operators = [ref]
    for _ in range(result_level):
        curr_operators = _gen_cluster_product(curr_operators, excitors)
        all_operators.append(deepcopy(curr_operators))
    return all_operators


def cc_to_ci_coeff(excitors, ci_index, result_level, err=False):
    """Convert HANDE CC amplitude to CI coefficient.
    Currently this is done brutally (and inefficiently), by adding all products
    of CC amplitudes up to the desired power. The sign and whether a product is
    valid is dealt with by the Cluster object.
    """
    ci_coeff = np.zeros(len(ci_index))
    ci_err = np.zeros(len(ci_index))
    all_operators = gen_cluster_operators(excitors, result_level)
    for prodpower, cluster_list in enumerate(all_operators):
        factor = 1/np.math.factorial(prodpower)
        for term in cluster_list:
            ci_coeff[ci_index[term.excitor]] += factor * term.amplitude
            ci_err[ci_index[term.excitor]] =\
                np.linalg.norm([ci_err[ci_index[term.excitor]], term.err])
    if err:
        return ci_coeff, ci_err
    return ci_coeff


def calc_proj_energy(hmat, ci_coeff, ci_index, ref, result_level, ci_err=None):
    """
    Calculate the projected energies onto each determinant with the CI
    coefficient information.
    """
    proj_energy = np.dot(hmat, ci_coeff) / ci_coeff
    proj_corr = proj_energy - hmat[0][0]
    if ci_err is not None:
        abs_err = new_calc_proje_err(hmat, ci_coeff, ci_err)
        # frac_err = calc_proje_err(hmat, ci_coeff, ci_err)
        # abs_err = abs(proj_corr) * frac_err
    ret_dict = {}
    for excitorid, _ in ci_index.items():
        excitor = Excitor(excitorid, ci_coeff[ci_index[excitorid]])
        excitor.decompose(ref)
        if excitor.exlevel <= result_level:
            try:
                ret_dict[excitorid] = (excitor.exlevel, excitor.amplitude,
                                       proj_corr[ci_index[excitorid]],
                                       abs_err[ci_index[excitorid]])
            except NameError:
                ret_dict[excitorid] = (excitor.exlevel, excitor.amplitude,
                                       proj_corr[ci_index[excitorid]])
    return ret_dict


def new_calc_proje_err(hmat, ci_coeff, ci_err):
    """
    Calculate the error in projected energy.
    """
    hmat0 = deepcopy(hmat)
    for lineid, _ in enumerate(hmat):
        hmat0[lineid][lineid] = 0
    cic_inv = 1 / ci_coeff
    cic_ratio = np.outer(cic_inv, ci_coeff)
    cic_rel_err_sq = (ci_err / ci_coeff)**2.
    cic_ratio_err = deepcopy(cic_ratio)
    for row in range(len(cic_ratio)):
        for col in range(len(cic_ratio)):
            cic_ratio_err[row][col] *= np.sqrt(cic_rel_err_sq[row] +
                                               cic_rel_err_sq[col])
    # sumHijnj = np.sum(hmat0 * cic_ratio, axis=1)

    cic_ratio_err = np.nan_to_num(cic_ratio_err)
    err_sumHijnj = np.linalg.norm(hmat0 * cic_ratio_err, axis=1)

    return err_sumHijnj


def separate_projected_e(hamilfile, wfnfile, rsfiles, level, refpop=None):
    """
    Generate projected energies onto all determinants of all files, and
    return in a 2D array.

    Returns:
        2D array of projected energies, indexed with (determinant, fileid)
    """
    hmat = read_hmat(hamilfile)
    ci_index = read_fci_space(wfnfile)
    result = np.empty((len(ci_index), len(rsfiles)))
    result[:] = np.nan
    for rsid, rsfile in enumerate(rsfiles):
        excitors = read_excitor(rsfile, refpop)
        ref = deepcopy(excitors[0]).excitor
        ci_coeff = cc_to_ci_coeff(excitors, ci_index, level)
        proj_energy_dict = calc_proj_energy(hmat, ci_coeff, ci_index, ref, level)
        for key, val in proj_energy_dict.items():
            result[ci_index[key], rsid] = val[2]
    return result


def separate_excitors(wfnfile, rsfiles):
    """
    Read all excitors and arrange in the same sequence as Hamiltonian matrix.
    """
    ci_index = read_fci_space(wfnfile)
    result = np.zeros((len(ci_index), len(rsfiles)))
    result[:] = np.nan
    for rsid, rsfile in enumerate(rsfiles):
        excitors = read_excitor(rsfile)
        for excitor in excitors:
            result[ci_index[excitor.excitor], rsid] = excitor.amplitude
    return result
