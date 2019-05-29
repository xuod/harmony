import os
import pymaster as nmt
import scipy.stats
from sklearn.covariance import GraphicalLassoCV
import numpy as np

try:
    FileNotFoundError
except NameError:
    #py2
    FileNotFoundError = IOError

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def compute_master(f_a, f_b, wsp, clb=None) :
    cl_coupled=nmt.compute_coupled_cell(f_a,f_b)
    cl_decoupled=wsp.decouple_cell(cl_coupled,cl_bias=clb)
    return cl_decoupled


def get_chi2(obs, randoms, smooth=False, return_pval=False):
    if smooth:
        model = GraphicalLassoCV(cv=5)
        model.fit(randoms)
        cov = model.covariance_
    else:
        cov = np.cov(randoms, rowvar=False)

    def calc_chi2(x, cov, xmean=None):
        if xmean is not None :
            y = x - xmean
        else :
            y = x
        icov = np.linalg.inv(cov)
        return np.dot(y.T, np.dot(icov, y))

    chi2 = calc_chi2(obs, cov, np.mean(randoms, axis=0))
    if return_pval:
        pval = scipy.stats.chi2.sf(chi2, df=obs.shape[0])
        return chi2, pval
    else:
        return chi2


def log_ell_bins(lmin, lmax, n_ell_bins):
    ell = np.arange(lmin,lmax+1)
    bins_ell = np.logspace(np.log10(lmin), np.log10(lmax), n_ell_bins+1)
    # Avoid int to float to int rounding errors...
    eps = 1e-5
    bins_ell[0] = lmin-eps
    bins_ell[-1] = lmax+eps

    # Check that all bins have at least one multipole
    assert np.all(np.histogram(ell, bins_ell)[0] > 0)

    # Digitize and make list
    bpws = np.digitize(ell, bins_ell, right=False)
    ells = []
    for ibin in range(1,n_ell_bins+1):
        w = (bpws == ibin)
        if np.sum(w)>0:
            ells.append(ell[w])
    
    return ells


def make_nmtbin_logspaced(nside, lmin, lmax, n_ell_bins):
    ells = log_ell_bins(lmin, lmax, n_ell_bins)
    bpws = np.concatenate([i*np.ones(len(x), dtype=int) for i,x in enumerate(ells)])
    ells = np.concatenate(ells)
    return nmt.NmtBin(nside, ells=ells, bpws=bpws, weights=np.ones(len(ells)))