import healpy as hp
import pymaster as nmt
from tqdm.auto import tqdm, trange
from astropy.io import fits
import os, sys
# sys.path.insert(0, os.path.join(os.environ['HOME'],'codes/castor'))
# sys.path.insert(0, os.path.join(os.environ['HOME'],'Cosmo/codes/castor'))
import castor as ca
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing
from .utils import *
import numpy as np
import logging
import scipy

class Observable(object):
    def __init__(self, config, nside, mode, nzbins, obs_name, map_names, nproc=0):
        self.config = config
        self.name = config.name
        self.nside = nside
        self.npix = hp.nside2npix(nside)

        self.map_names = map_names
        self.obs_name = obs_name
        self.mode = mode
        # self.nzbins = nzbins

        if type(nzbins) == int:
            self.zbins = list(range(nzbins))
            self.nzbins = nzbins
        else:
            self.zbins = nzbins
            self.nzbins = len(nzbins)

        self.maps = {} # to be organized as maps[redshift_bin][map_name]
        self.masks = {} # to be organized as maps[redshift_bin][map_name] # NO !
        self.masks_apo = {} # to be organized as maps[redshift_bin][map_name] # NO !

        for i in self.zbins:
            self.masks[i] = None
            self.masks_apo[i] = None
            self.maps[i] = {}
            for name in map_names:
                self.maps[i][name] = None

        self.nproc = nproc
        if nproc > 1:
            self.pool = multiprocessing.Pool(nproc)

        self.templates = 12345 # this will cause error if templates is called without the function _get_templates_array first
        self.template_dir = None

    # def load_catalogs(self):
    #     print('Method load_catalogs not implemented, nothing to do.')

    def make_maps(self):
        print('Method make_maps not implemented, nothing to do.')

    def save_maps(self):
        maps_dir = os.path.join(self.config.path_maps, self.name)
        make_directory(maps_dir)
        for ibin in tqdm(self.zbins, desc='{}.save_maps'.format(self.obs_name)):
            hp.write_map(os.path.join(maps_dir, '{}_{}_{}_nside{}_bin{}.fits'.format('mask', self.config.name, self.mode, self.nside, ibin)), self.masks[ibin], overwrite=True)
            for map_name in self.map_names:
                hp.write_map(os.path.join(maps_dir, '{}_{}_{}_nside{}_bin{}.fits'.format(map_name, self.config.name, self.mode, self.nside, ibin)), self.maps[ibin][map_name], overwrite=True)

    def load_maps(self):
        maps_dir = os.path.join(self.config.path_maps, self.name)
        for ibin in tqdm(self.zbins, desc='{}.load_maps'.format(self.obs_name)):
            # self.maps[ibin] = {}
            self.masks[ibin] = hp.read_map(os.path.join(maps_dir, '{}_{}_{}_nside{}_bin{}.fits'.format('mask', self.config.name, self.mode, self.nside, ibin)), verbose=False)
            for map_name in self.map_names:
                self.maps[ibin][map_name] = hp.read_map(os.path.join(maps_dir, '{}_{}_{}_nside{}_bin{}.fits'.format(map_name, self.config.name, self.mode, self.nside, ibin)), verbose=False)

    def plot_maps(self):
        make_directory(self.config.path_figures+'/'+self.name)
        for ibin in tqdm(self.zbins, desc='{}.plot_maps'.format(self.obs_name)):
            for map_name in self.map_names:
                hp.mollview(self.maps[ibin][map_name], title='{} (bin {})'.format(map_name, ibin))
                figfile = os.path.join(self.config.path_figures, self.name, '{}_{}_{}_nside{}_bin{}.png'.format(map_name, self.config.name, self.mode, self.nside, ibin))
                plt.savefig(figfile, dpi=300)
                plt.show()

    def make_masks_apo(self, hm):
        self.masks_apo = {}
        for ibin in tqdm(self.zbins, desc='{}.make_masks_apo'.format(self.obs_name)):
            # self.masks_apo[ibin] = {}
            # for map_name in self.map_names:
            self.masks_apo[ibin] = nmt.mask_apodization(self.masks[ibin], aposize=hm.aposize, apotype=hm.apotype)

    def _get_templates_array(self):
        if self.template_dir is None:
            return None
        else:
            if self.templates == 12345:
                templates = []
                for key, temp in self.templates_dir:
                    templates.append(temp)
                self.templates = np.expand_dims(np.array(templates), axis=1)
                return self.templates
            else:
                return self.templates

    def _init_templates(self):
        if self.templates is 12345:
            self.template_dir = {}
            # self.templates = []

    def load_all_templates_from_dir(self, templates_dir):
        self._init_templates()

        template_names = os.listdir(templates_dir)
        template_names.sort()
        for filename in tqdm(template_names, desc='{}.load_all_templates_from_dir'.format(self.obs_name)):
            temp = hp.read_map(os.path.join(templates_dir, filename), verbose=False)
            self.template_dir[filename] = temp
            # self.templates.append(temp)

        # self.templates = np.array(self.templates)
        # self.templates = np.expand_dims(self.templates, axis=1)

    def load_template(self, filename, tempname):
        self._init_templates()
        self.template_dir[tempname] =  hp.read_map(filename, verbose=False)

    def load_DES_templates(self, templates_dir, bands=['g', 'r', 'i', 'z']):
        self._init_templates()

        self.bands = bands
        self.syst = {}
        self.syst['Air mass'] = 'AIRMASS.WMEAN_EQU'
        self.syst['Seeing'] = 'FWHM.WMEAN_EQU'
        self.syst['Zero point residuals'] = 'SIGMA_MAG_ZERO.QSUM_EQU'
        self.syst['Sky variance'] = 'SKYVAR_WMEAN_EQU'
        self.syst['Effective exposure time (mean)'] = 'T_EFF.WMEAN_EQU'
        self.syst['Effective exposure time (sum)'] = 'T_EFF_EXPTIME.SUM_EQU'

        # self.template_dir = {}
        # self.templates = []

        for band in self.bands:
            for key, name in self.syst.items():
                tempname = 'y3a2_{}_o.4096_t.32768_{}._nside{}.fits'.format(band, name, self.nside)
                temp = hp.read_map(os.path.join(templates_dir, tempname), verbose=False)
                self.template_dir['%s [%s band]'%(key, band)] = temp
                # self.templates.append(temp)

        # self.templates = np.array(self.templates)
        # self.templates = np.expand_dims(self.templates, axis=1)

    def load_PSF(self, hm, PSF_dir, make_fields=True):
        psf_maps = {}

        keys = ['obs_e1', 'obs_e2', 'piff_e1', 'piff_e2', 'mask']
        for k in keys:
            psf_maps[k] = hp.read_map(os.path.join(PSF_dir, 'PSF_%s_nside%i.fits'%(k, self.nside)), verbose=False)

        self.psf_maps = {}
        self.psf_maps['obs'] = [psf_maps['obs_e1'], -1.0*psf_maps['obs_e2']]
        self.psf_maps['piff'] = [psf_maps['piff_e1'], -1.0*psf_maps['piff_e2']]
        self.psf_maps['res'] = [psf_maps['obs_e1']-psf_maps['piff_e1'], -1.0*(psf_maps['obs_e2']-psf_maps['piff_e2'])]

        self.psf_mask_apo = nmt.mask_apodization(psf_maps['mask'], aposize=hm.aposize, apotype=hm.apotype)

        self.psf_fields = {}
        for k in self.psf_maps.keys():
            if make_fields:
                self.psf_fields[k]  = nmt.NmtField(self.psf_mask_apo, self.psf_maps[k], purify_e=hm.purify_e, purify_b=hm.purify_b)
            else:
                self.psf_fields[k]  = None
        # self.psf_fields['obs']  = nmt.NmtField(psf_mask_apo, [self.psf_maps['obs_e1'], -1.0*self.psf_maps['obs_e2']], purify_e=hm.purify_e, purify_b=hm.purify_b)
        # self.psf_fields['piff'] = nmt.NmtField(psf_mask_apo, [self.psf_maps['piff_e1'], -1.0*self.psf_maps['piff_e2']], purify_e=hm.purify_e, purify_b=hm.purify_b)
        # self.psf_fields['res']  = nmt.NmtField(psf_mask_apo, [self.psf_maps['obs_e1']-self.psf_maps['piff_e1'], -1.0*(self.psf_maps['obs_e1']-self.psf_maps['piff_e2'])], purify_e=hm.purify_e, purify_b=hm.purify_b)
        #
        # return psf_maps

    def get_field(self, hm, ibin, include_templates=True):
        raise NotImplementedError

    def get_randomized_fields(self, hm, ibin, nsamples=1):
        raise NotImplementedError

    def get_randomized_map(self, ibin):
        raise NotImplementedError

    def _compute_auto_cls(self, hm, ibin, nrandom=0, save=True):
        raise NotImplementedError

    def plot_cls(self, hm, cls, nrows, ncols, figname='', titles=None, ylabels=None, showy0=False, symy0=False, chi2method=None):
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, 3*nrows))
        axes = axes.reshape((nrows,ncols))

        ell = hm.b.get_effective_ells()

        chi2 = {}
        for i in range(nrows):
            for j in range(ncols):
                ax = axes[i,j]
                if showy0:
                    ax.axhline(y=0, c='0.8', lw=1)
                y = cls[(i,j)]
                nrandom = y['random'].shape[0]
                for r in range(nrandom):
                    ax.plot(ell, y['random'][r], c='r', alpha=max(0.01, 1./nrandom))
                if chi2method is not None:
                    _chi2 = get_chi2(y['true'], y['random'], smooth=(chi2method=='smooth'))
                    label = '$\\chi^2_{{{:}}} = {:.2f}$ ($p={:.2g}$)'.format(len(ell), _chi2, scipy.stats.chi2.sf(_chi2, df=hm.b.get_n_bands()))
                    chi2[(i,j)] = _chi2
                else:
                    label = None
                ax.plot(ell, y['true'], label=label, c='b')
                if titles is not None:
                    ax.set_title(titles[(i,j)], fontsize=8)
                ax.set_xlabel('$\\ell$')
                ax.set_xlim(0, hm.b.lmax)
                if ylabels is not None:
                    ax.set_ylabel(ylabels[(i,j)])
                if symy0:
                    vmax = max(np.abs(ax.get_ylim()))
                    ax.set_ylim(-vmax,+vmax)
                if chi2method is not None:
                    ax.legend(loc=1)
        plt.tight_layout()

        make_directory(self.config.path_figures+'/'+self.name)
        figfile = os.path.join(self.config.path_figures, self.name, 'cls_{}_{}_{}_{}_nside{}.png'.format(figname, self.obs_name, self.config.name, self.mode, self.nside))
        plt.savefig(figfile, dpi=300)

        if chi2method is not None:
            return chi2

    def _compute_cross_template_cls(self, hm, ibin, nrandom=0, save=True):
        raise NotImplementedError

    def _compute_cross_PSF_cls(self, hm, ibin, nrandom=0, save=True):
        raise NotImplementedError
