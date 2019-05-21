import healpy as hp
import pymaster as nmt
from tqdm.auto import tqdm, trange
from astropy.io import fits
import os, sys
import castor as ca
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing
from .utils import *
import pickle
import numpy as np

class Harmony(object):
    def __init__(self, config, nside, aposize=2.0, apotype='C1', purify_e=False, purify_b=False, nlb=32, lmax=None, b=None, nproc=0):
        self.config = config
        self.name = config.name
        self.nside = nside

        self.aposize = aposize
        self.apotype = apotype
        self.purify_e = purify_e
        self.purify_b = purify_b

        if b is None:
            self.lmax = lmax
            self.nlb = nlb
            self.b = nmt.NmtBin(self.nside, nlb=nlb, lmax=lmax)
        else:
            self.b = b
            self.lmax = b.lmax
            # self.nlb = b.nlb
        self.ell = self.b.get_effective_ells()

        self.cls = {}
        self.cls['ell'] = self.ell

        self.nproc = nproc
        if nproc > 1:
            self.pool = multiprocessing.Pool(nproc)

    def check_cls_obs(self, obs1, obs2):
        key = (obs1.obs_name, obs2.obs_name)
        if key not in self.cls.keys():
            self.cls[key] = {}
        else:
            print("Replacing cls[%s]".format(str(key)))

    def compute_cross_cls(self, obs1, obs2, i1, i2, save=True):
        self.check_cls_obs(obs1, obs2)

        field1 = obs1.get_field(self, i1)
        field2 = obs2.get_field(self, i2)

        self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = nmt.compute_full_master(field1, field2, self.b)

        if save:
            self.save_cls()

    def compute_all_cls(self, obs1, obs2=None, save=True):
        if obs2 is None:
            same_obs = True
            obs2 = obs1

        self.check_cls_obs(obs1, obs2)

        for i1 in tqdm(obs1.zbins, desc='Harmony.compute_all_cls [obs1:{}, obs2={}]'.format(obs1.obs_name, obs2.obs_name)):
            field1 = obs1.get_field(self, i1)
            for i2 in obs2.zbins:
                if (i2,i1) in self.cls[(obs1.obs_name, obs2.obs_name)].keys() and same_obs:
                        self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = self.cls[(obs1.obs_name, obs2.obs_name)][(i2,i1)]
                else:
                    field2 = obs2.get_field(self, i2)
                    self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = nmt.compute_full_master(field1, field2, self.b)

        if save:
            self.save_cls()

    def compute_auto_cls(self, obs, nrandom=0, save=True):
        self.check_cls_obs(obs, obs)

        for ibin in tqdm(obs.zbins, desc='Harmony.compute_auto_cls [obs:{}]'.format(obs.obs_name)):
            self.cls[(obs.obs_name, obs.obs_name)][(ibin,ibin)] = obs._compute_auto_cls(self, ibin, nrandom=nrandom, save=save)

            if save:
                self.save_cls()

    def compute_cross_template_cls(self, obs, nrandom, save=True):
        for tempname in obs.template_dir.keys():
            key = (obs.obs_name, tempname)
            if key not in self.cls.keys():
                self.cls[key] = {}
            else:
                print("Replacing cls[%s]".format(str(key)))

        for ibin in tqdm(obs.zbins, desc='Harmony.compute_cross_template_cls [obs:{}]'.format(obs.obs_name)):
            obs._compute_cross_template_cls(self, ibin, nrandom=nrandom)

            if save:
                self.save_cls()

    def compute_cross_PSF_cls(self, obs, nrandom, save=True):
        for tempname in obs.psf_maps.keys():
            key = (obs.obs_name, tempname)
            if key not in self.cls.keys():
                self.cls[key] = {}
            else:
                print("Replacing cls[%s]".format(str(key)))

        for ibin in tqdm(obs.zbins, desc='Harmony.compute_cross_PSF_cls [obs:{}]'.format(obs.obs_name)):
            obs._compute_cross_PSF_cls(self, ibin, nrandom=nrandom)

            if save:
                self.save_cls()

    def save_cls(self):
        make_directory(self.config.path_output+'/'+self.name)
        filename = os.path.join(self.config.path_output, self.name, 'cls_{}_nside{}.pickle'.format(self.config.name, self.nside))
        pickle.dump(self.cls, open(filename, mode='wb'))

    def load_cls(self):
        filename = os.path.join(self.config.path_output, self.name, 'cls_{}_nside{}.pickle'.format(self.config.name, self.nside))
        try:
            self.cls = pickle.load(open(filename, mode='rb'))
        except FileNotFoundError:
            print("Cls file does not exists: {}".format(filename))
