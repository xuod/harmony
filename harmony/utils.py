import os
import pymaster as nmt
import scipy.stats
from sklearn.covariance import GraphicalLassoCV

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

def get_chi2_smoothcov(obs, randoms):
    model = GraphicalLassoCV(cv=5)
    model.fit(randoms)
    return ca.maths.calc_chi2(obs, model.covariance_, np.mean(randoms, axis=0))
