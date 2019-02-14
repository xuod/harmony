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
