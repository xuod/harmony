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

class Observable(object):
    def __init__(self, config, nside, mode, nzbins, obs_name, map_names, nproc=0):
        self.config = config
        self.name = config.name
        self.nside = nside
        self.npix = hp.nside2npix(nside)

        self.nproc = nproc
        if nproc > 1:
            self.pool = multiprocessing.Pool(nproc)

        self.map_names = map_names
        self.obs_name = obs_name
        self.mode = mode
        self.nzbins = nzbins

        self.maps = {} # to be organized as maps[redshift_bin][map_name]
        self.masks = {} # to be organized as maps[redshift_bin][map_name] # NO !
        self.masks_apo = {} # to be organized as maps[redshift_bin][map_name] # NO !
        for i in range(nzbins):
            self.masks[i] = None
            self.masks_apo[i] = None
            self.maps[i] = {}
            for name in map_names:
                self.maps[i][name] = None

        self.nproc = nproc
        if nproc > 1:
            self.pool = multiprocessing.Pool(nproc)

        self.templates = None

    # def load_catalogs(self):
    #     print('Method load_catalogs not implemented, nothing to do.')

    def make_maps(self):
        print('Method make_maps not implemented, nothing to do.')

    def save_maps(self):
        maps_dir = os.path.join(self.config.path_maps, self.name)
        make_directory(maps_dir)
        for ibin in trange(self.nzbins, desc='{}.save_maps'.format(self.obs_name)):
            hp.write_map(os.path.join(maps_dir, '{}_{}_{}_nside{}_bin{}.fits'.format('mask', self.config.name, self.mode, self.nside, ibin)), self.masks[ibin], overwrite=True)
            for map_name in self.map_names:
                hp.write_map(os.path.join(maps_dir, '{}_{}_{}_nside{}_bin{}.fits'.format(map_name, self.config.name, self.mode, self.nside, ibin)), self.maps[ibin][map_name], overwrite=True)

    def load_maps(self):
        maps_dir = os.path.join(self.config.path_maps, self.name)
        for ibin in trange(self.nzbins, desc='{}.load_maps'.format(self.obs_name)):
            # self.maps[ibin] = {}
            self.masks[ibin] = hp.read_map(os.path.join(maps_dir, '{}_{}_{}_nside{}_bin{}.fits'.format('mask', self.config.name, self.mode, self.nside, ibin)), verbose=False)
            for map_name in self.map_names:
                self.maps[ibin][map_name] = hp.read_map(os.path.join(maps_dir, '{}_{}_{}_nside{}_bin{}.fits'.format(map_name, self.config.name, self.mode, self.nside, ibin)), verbose=False)

    def plot_maps(self):
        make_directory(self.config.path_figures+'/'+self.name)
        for ibin in trange(self.nzbins, desc='{}.plot_maps'.format(self.obs_name)):
            for map_name in self.map_names:
                hp.mollview(self.maps[ibin][map_name], title='{} (bin {})'.format(map_name, ibin))
                figfile = os.path.join(self.config.path_figures, self.name, '{}_{}_{}_nside{}_bin{}.png'.format(map_name, self.config.name, self.mode, self.nside, ibin))
                plt.savefig(figfile, dpi=300)
                plt.show()

    def make_masks_apo(self, hm):
        self.masks_apo = {}
        for ibin in trange(self.nzbins, desc='{}.make_masks_apo'.format(self.obs_name)):
            # self.masks_apo[ibin] = {}
            # for map_name in self.map_names:
            self.masks_apo[ibin] = nmt.mask_apodization(self.masks[ibin], aposize=hm.aposize, apotype=hm.apotype)


    def load_all_templates_from_dir(self, templates_dir):
        self.template_names = os.listdir(templates_dir)
        self.template_names.sort()
        for filename in tqdm(self.template_names, desc='{}.load_all_templates_from_dir'.format(self.obs_name)):
            temp = hp.read_map(os.path.join(templates_dir, filename) ,verbose=False)
            self.template_dir[filename] = temp
            self.templates.append(temp)

        self.templates = np.array(self.templates)
        self.templates = np.expand_dims(self.templates, axis=1)

    def load_DES_templates(self, templates_dir):
        self.bands = ['g', 'r', 'i', 'z']
        self.syst = {}
        self.syst['Air mass'] = 'AIRMASS.WMEAN_EQU'
        self.syst['Seeing'] = 'FWHM.WMEAN_EQU'
        self.syst['Zero point residuals'] = 'SIGMA_MAG_ZERO.QSUM_EQU'
        self.syst['Sky variance'] = 'SKYVAR_WMEAN_EQU'
        self.syst['Effective exposure time (mean)'] = 'T_EFF.WMEAN_EQU'
        self.syst['Effective exposure time (sum)'] = 'T_EFF_EXPTIME.SUM_EQU'

        self.template_dir = {}
        self.templates = []

        for band in self.bands:
            for key, name in self.syst.items():
                tempname = 'y3a2_{}_o.4096_t.32768_{}.fits'.format(band, name)
                temp = hp.read_map(os.path.join(templates_dir, tempname), verbose=False)
                self.template_dir['%s [%s band]'%(key, band)] = temp
                self.templates.append(temp)

        self.templates = np.array(self.templates)
        self.templates = np.expand_dims(self.templates, axis=1)

    def get_field(self, hm, ibin, include_templates=True):
        raise NotImplementedError

    def get_randomized_fields(self, hm, ibin, nsamples=1):
        raise NotImplementedError

    def get_randomized_map(self, ibin):
        raise NotImplementedError

    def _compute_auto_cls(self, hm, ibin, nrandom=0, save=True):
        raise NotImplementedError

    def plot_auto_cls(self, hm, *args, **kwargs):
        raise NotImplementedError

    def compute_cross_cls_templates(self, hm, nrandom=0):
        template_fields = {}
        for key, temp in self.template_dir.items():
            mask = np.logical_not((temp == hp.UNSEEN) | (temp == 0.0)) # kinda dangerous...
            template_fields[key] = nmt.NmtField(mask, [temp])

        fields = {}
        fields_r = {}
        for ibin in range(self.nzbins):
            fields[ibin] = self.get_field(hm, ibin, include_templates=False)
            if nrandom > 0:
                fields_r[ibin] = self.get_randomized_fields(hm, ibin, nrandom)

        cls = {}

        for ibin in trange(self.nzbins):
            for key, f_temp in template_fields.items():
                cls[(ibin, key)] = {}
                wsp = nmt.NmtWorkspace()
                wsp.compute_coupling_matrix(fields[ibin], f_temp, hm.b)

                cls[(ibin, key)]['true'] = compute_master(fields[ibin], f_temp, wsp)

                if nrandom > 0:
                    cls[(ibin, key)]['random'] = []
                    for i in range(nrandom):
                        cls[(ibin,key)]['random'].append(compute_master(fields_r[ibin][i], f_temp, wsp))
                    cls[(ibin,key)]['random'] = np.array(cls[(ibin, key)]['random'])

        return cls
