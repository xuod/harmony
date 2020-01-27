import os
import pymaster as nmt
import scipy.stats
from sklearn.covariance import GraphicalLassoCV
import numpy as np
import healpy as hp
import castor as ca
import warnings
from collections.abc import Iterable
from tqdm.auto import tqdm, trange

try:
    FileNotFoundError
except NameError:
    #py2
    FileNotFoundError = IOError

def prog(verbose):
    if verbose:
        def f(x, *args, **kwargs):
            if isinstance(x, Iterable):
                return tqdm(x, *args, dynamic_ncols=True, **kwargs)
            elif int(x)==x:
                return trange(x, *args, dynamic_ncols=True, **kwargs)
            else:
                raise NotImplementedError

    else:
        def f(x, *args, **kwargs):
            if isinstance(x, Iterable):
                return x
            elif int(x)==x:
                return range(x)
            else:
                raise NotImplementedError
    return f

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def compute_master(f_a, f_b, wsp, clb=None) :
    cl_coupled=nmt.compute_coupled_cell(f_a,f_b)
    cl_decoupled=wsp.decouple_cell(cl_coupled,cl_bias=clb)
    return cl_decoupled

def hpunseen2zero(map_in):
    map_out = np.copy(map_in)
    map_out[map_out==hp.UNSEEN] = 0.
    return map_out

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


# def log_ell_bins(lmin, lmax, n_ell_bins):
#     ell = np.arange(lmin,lmax+1)
#     bins_ell = np.logspace(np.log10(lmin), np.log10(lmax), n_ell_bins+1)
#     # Avoid int to float to int rounding errors...
#     eps = 1e-5
#     bins_ell[0] = lmin-eps
#     bins_ell[-1] = lmax+eps

#     # Check that all bins have at least one multipole
#     assert np.all(np.histogram(ell, bins_ell)[0] > 0)

#     # Digitize and make list
#     bpws = np.digitize(ell, bins_ell, right=False)
#     ells = []
#     for ibin in range(1,n_ell_bins+1):
#         w = (bpws == ibin)
#         if np.sum(w)>0:
#             ells.append(ell[w])
    
#     return ells

# def make_nmtbin_logspaced(nside, lmin, lmax, n_ell_bins, f_ell=None, add_extra_bin=False, **kwargs):
#     if add_extra_bin:
#         ells = log_ell_bins(lmin, int(lmax*np.exp((np.log(lmax)-np.log(lmin))/n_ell_bins)), n_ell_bins+1)
#     else:
#         ells = log_ell_bins(lmin, lmax, n_ell_bins)
#     bpws = np.concatenate([i*np.ones(len(x), dtype=int) for i,x in enumerate(ells)])
#     ells = np.concatenate(ells)
#     if f_ell is not None:
#         if f_ell=='pixwin':
#             f_ell = 1./hp.pixwin(nside)[lmin:lmax+1]**2
#         else:
#             assert type(f_ell) == np.ndarray
#     return nmt.NmtBin(nside, ells=ells, bpws=bpws, weights=np.ones(len(ells)), lmax=lmax, f_ell=f_ell, **kwargs)

def make_nmtbin(nside, lmin, lmax, n_ell_bins, bin_func=np.linspace, f_ell=None, verbose=False, b_lmax=None, bins_ell=None, **kwargs):
    # Define ell range
    ells = np.arange(lmin, lmax+1)

    # Make bin limits
    if bins_ell is not None:
        warnings.warn("Warning: using provided `bins_ell` instead of computing it from `bin_func`.")
        assert bins_ell[0]==lmin and bins_ell[-1]==lmax
    else:
        bins_ell = bin_func(lmin, lmax, n_ell_bins+1).astype(float)

    # Get bands
    bpws = np.digitize(ells.astype(float), bins_ell) - 1
    bpws[0] = 0 # fixes bin edge issues...
    bpws[-1] = n_ell_bins-1

    # Make sure each multipole is assigned to one of the n_ell_bins bins
    assert np.all(np.unique(bpws) == np.arange(n_ell_bins))

    # Multiplicative factor
    if f_ell is not None:
        if f_ell=='pixwin':
            f_ell = 1./hp.pixwin(nside)[lmin:lmax+1]**2
        else:
            assert type(f_ell) == np.ndarray
            assert len(f_ell) == len(ells)
        
    b = nmt.NmtBin(nside, ells=ells, bpws=bpws, weights=np.ones(len(ells)), f_ell=f_ell, lmax=b_lmax, **kwargs)

    if b.get_n_bands() != n_ell_bins:
        warnings.warn("Warning: for some reason, b.get_n_bands() != n_ell_bins")

    if verbose:
        for i in range(b.get_n_bands()):
            print("Bin {:4d} = [{:5d} - {:5d}]".format(i, b.get_ell_list(i)[0], b.get_ell_list(i)[-1]))

    return b

def make_nmtbin_logspaced(nside, lmin, lmax, n_ell_bins, f_ell=None, **kwargs):
    return make_nmtbin(nside, lmin, lmax, n_ell_bins, bin_func=np.geomspace, f_ell=f_ell, **kwargs)

def make_nmtbin_linspaced(nside, lmin, lmax, n_ell_bins, f_ell=None, **kwargs):
    return make_nmtbin(nside, lmin, lmax, n_ell_bins, bin_func=np.linspace, f_ell=f_ell, **kwargs)

def load_cosmosis_cl(dir_path, ell_interp=None, starts_at_1=True, imax=10, symmetrize=False):
    offset = int(starts_at_1)

    cell = {}
    ell_cosmosis = np.loadtxt(os.path.join(dir_path, 'ell.txt'))
    if ell_interp is not None:
        cell['ell'] = ell_interp
    else:
        cell['ell'] = ell_cosmosis

    for i in range(imax):
        for j in range(imax):
            try:
                temp = np.loadtxt(os.path.join(dir_path, 'bin_{}_{}.txt'.format(i+offset,j+offset)))
                if ell_interp is not None:
                    cell[(i,j)] = ca.maths.interp_loglog(ell_interp, ell_cosmosis, temp, logx=True, logy=True)
                else:
                    cell[(i,j)] = temp
                if symmetrize:
                    cell[(j,i)] = cell[(i,j)]
            except OSError:
                pass
    
    return cell

# Useless, just saving it for the method
# class PickableSWIG:
#     def __setstate__(self, state):
#         self.__init__(*state['args'], **state['kwargs'])

#     def __getstate__(self):
#         return {'args': self.args, 'kwargs': self.kwargs}

# class PickableNmtField(nmt.NmtField, PickableSWIG):
#     def __init__(self, *args, **kwargs):
#         self.args = args
#         self.kwargs = kwargs
#         nmt.NmtField.__init__(self, *args, **kwargs)