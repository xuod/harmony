from .observable import Observable
from .utils import hpunseen2zero
import numba
import twopoint
from astropy.io import fits
import os
import healpy as hp
import twopoint
import numpy as np
import castor as ca

class Galaxy(Observable):
    def __init__(self, config, nside, mode, nzbins, mask_mode, data_dir='../data', mask_dir='../masks', get_count_weights_from=None, dry_run=False, density_convention=2, completeness_cut=0.8, *args, **kwargs):

        self.obs_name = 'galaxy_density'
        self.map_names = ['count', 'density', 'completeness']

        super(Galaxy, self).__init__(config, nside, mode, nzbins, self.obs_name, self.map_names, *args, **kwargs)

        self.spin = 0
        self.kernel = 'nz_lens'
        self.type = twopoint.Types.galaxy_position_fourier

        self.data_dir = data_dir
        self.mask_dir = mask_dir

        assert density_convention in [1,2]
        self.density_convention = density_convention
        self.completeness_cut = completeness_cut

        assert mask_mode in ['binary', 'weights', 'inverse_weights', 'completeness', 'weights*completeness', 'inverse_weights*completeness']
        self.mask_mode = mask_mode

        assert get_count_weights_from in [None, 'mask', 'catalog']
        self.get_count_weights_from = get_count_weights_from

        load_catalog_weights=(get_count_weights_from=='catalog')
        load_weights_maps = ('weights' in mask_mode) or (get_count_weights_from=='mask')

        self.cats = {}

        if not dry_run:
            if mode=='redmagic_Y3':
                self.init_redmagic_Y3(self.nzbins, load_catalog_weights=load_catalog_weights, load_weights_maps=load_weights_maps)
            else:
                raise NotImplementedError

    def init_redmagic_Y3(self, nzbins, load_catalog_weights, load_weights_maps):

        for ibin in self.prog(self.zbins, desc='Galaxy.init_redmagic_Y3'):
            # Loading catalog
            temp = fits.open(os.path.join(self.data_dir, 'redmagic_bin{}.fits'.format(ibin)))[1].data
            self.cats[ibin] = {}
            self.cats[ibin]['ra'] = temp['ra']
            self.cats[ibin]['dec'] = temp['dec']

            # Load weights from catalog
            if load_catalog_weights:
                self.cats[ibin]['weight'] = temp['weight']

            # Loading weights and completeness map
            comp = hpunseen2zero(hp.read_map(os.path.join(self.data_dir, 'redmagic_bin{}_comp_nside{}.fits'.format(ibin, self.nside)), verbose=False, dtype=np.float64))
            if load_weights_maps:
                weights = hpunseen2zero(hp.read_map(os.path.join(self.data_dir, 'redmagic_bin{}_binary_nside{}.fits'.format(ibin, self.nside)), verbose=False, dtype=np.float64))
            else:
                weights = np.ones_like(comp)

            # Exclude regions where completeness is below a certain level
            if self.completeness_cut is not None:
                w = comp < self.completeness_cut
                comp[w] = 0.
            
            # Making sure to exclude regions where completeness is 0
            bool_mask = comp>0.
            weights[np.logical_not(bool_mask)] = 0.

            # Getting completeness
            self.maps[ibin]['completeness'] = comp
                        
            # Getting mask for NmtField
            if self.mask_mode == 'binary':
                self.masks[ibin] = (weights > 0.).astype(float)
            elif self.mask_mode == 'weights':
                self.masks[ibin] = weights
            elif self.mask_mode == 'inverse_weights':
                self.masks[ibin] = np.zeros_like(weights)
                self.masks[ibin][bool_mask] = 1./weights[bool_mask]
            elif self.mask_mode == 'completeness':
                self.masks[ibin] = comp
            elif self.mask_mode == 'weights*completeness':
                self.masks[ibin] = weights * comp
            elif self.mask_mode == 'inverse_weights*completeness':
                self.masks[ibin] = np.zeros_like(weights)
                self.masks[ibin][bool_mask] = comp[bool_mask]/weights[bool_mask]
            else:
                raise NotImplementedError

            if self.get_count_weights_from=='mask':
                self.cats[ibin]['weights_map'] = weights

    def init_catalog(self, ibin, ra, dec, mask, completeness=None, weight=None):
        if ibin in self.cats.keys():
            print("[init_catalog] Replacing catalog {}".format(ibin))
        self.cats[ibin] = {}

        assert len(ra)==len(dec)
        self.cats[ibin]['ra'] = ra
        self.cats[ibin]['dec'] = dec
        if weight is None:
            self.cats[ibin]['weight'] = np.ones(len(ra))
        else:
            self.cats[ibin]['weight'] = weight

        self.masks[ibin] = mask
        if completeness is None:
            self.maps[ibin]['completeness'] = (mask>0.).astype(float)
        else:
            self.maps[ibin]['completeness'] = completeness


    def make_maps(self, save=True, rotator=None):
        for ibin in self.prog(self.zbins, desc='Galaxy.make_maps'):
            cat = self.cats[ibin]

            if rotator is None:
                ra = cat['ra']
                dec = cat['dec']
            else:
                theta, phi = rotator(ca.cosmo.radec2thetaphi(cat['ra'], cat['dec']))
                ra, dec = ca.cosmo.thetaphi2radec(theta, phi)

            # get_count_weights_from in [None, 'mask', 'catalog']
            if self.get_count_weights_from is None:
                _, count, _ = ca.cosmo.make_healpix_map(ra, dec, None, self.nside,
                                            mask=None,
                                            weight=None,
                                            fill_UNSEEN=False, return_extra=False)

                density = ca.cosmo.count2density(count, mask=self.masks[ibin], completeness=self.maps[ibin]['completeness'], density_convention=self.density_convention)

            elif self.get_count_weights_from=='mask':
                cat = self.cats[ibin]
                ipix = hp.ang2pix(self.nside, (90.-dec)*np.pi/180.0, ra*np.pi/180.0)
                weights = cat['weights_map'][ipix]
                w = weights>0.
                count_w, count, _ = ca.cosmo.make_healpix_map(None, None,
                                            [np.ones(np.sum(w))], self.nside,
                                            mask=None,
                                            weight=[weights[w]], ipix=ipix[w],
                                            fill_UNSEEN=False, return_extra=False, mode='sum')

                density = ca.cosmo.count2density(count_w[0], mask=self.masks[ibin], completeness=self.maps[ibin]['completeness'], density_convention=self.density_convention)

            else: #self.get_count_weights_from=='catalog'
                count_w, count, _ = ca.cosmo.make_healpix_map(ra, dec,
                                            [np.ones_like(cat['dec'])], self.nside,
                                            mask=None,
                                            weight=[cat['weight']],
                                            fill_UNSEEN=False, return_extra=False, mode='sum')

                density = ca.cosmo.count2density(count_w[0], mask=self.masks[ibin], completeness=self.maps[ibin]['completeness'], density_convention=self.density_convention)

            self.maps[ibin]['count'] = count
            self.maps[ibin]['density'] = density

        if save:
            self.save_maps()

    def prepare_fields(self):
        out = {}
        for ibin in self.zbins:
            out[ibin] = [self.maps[ibin]['density']]
        return out

    def _get_info(self):
        import pandas as pd

        info = {}
        info['fsky'] = []
        info['fsky (effective)'] = []
        info['Ngal'] = []
        info['area (std)'] = []
        info['area (effective, std)'] = []
        info['nbar (std)'] = []
        info['nbar (effective, std)'] = []

        for i in self.zbins:
            bool_mask = self.masks[i]>0.
            fsky = np.sum(bool_mask) * 1./ len(bool_mask)
            fsky_eff = np.sum(self.maps[i]['completeness']) * 1./ len(bool_mask)
            area = 4.*np.pi * fsky
            area_eff = 4.*np.pi * fsky_eff
            Ngal = int(np.sum(self.maps[i]['count'][bool_mask]))
            # Ngal_weighted = np.average(self.maps[i]['count'], weights=np.sum(self.masks[i]))

            nbar = Ngal / area
            nbar_eff = Ngal / area_eff
            
            info['fsky'].append(fsky)
            info['fsky (effective)'].append(fsky_eff)
            info['area (std)'].append(area)
            info['area (effective, std)'].append(area_eff)
            info['Ngal'].append(Ngal)
            info['nbar (std)'].append(nbar)
            info['nbar (effective, std)'].append(nbar_eff)

        df = pd.DataFrame(index=self.zbins, data=info)
        return df
    
    def make_randomized_maps(self, ibin):
        Nobj = len(self.cats[ibin]['ra'])
        count = random_count(self.maps[ibin]['completeness'], Nobj)
        density = ca.cosmo.count2density(count, mask=self.masks[ibin], completeness=self.maps[ibin]['completeness'])

        return [density]

    def plot_auto_cls(self, hm, remove_Nl=False, **kwargs):
        cls = {}
        titles = {}
        ylabels = {}

        # idx_EB = [0, 1, 3]

        for i, zbin in enumerate(self.zbins):
            k = (0,i)
            titles[k] = ' [bin %i]'%(i+1)
            ylabels[k] = '$C_\\ell$'
            cls[k] = {}
            cls[k]['data'] = np.copy(hm.cls[(self.obs_name, self.obs_name)][zbin]['data'][0])
            cls_r = hm.cls[(self.obs_name, self.obs_name)][zbin]['random'][:,0,:]
            if remove_Nl:
                cls[k]['data'] -= np.mean(cls_r, axis=0)
            else:
                cls[k]['random'] = cls_r

        return self.plot_cls(hm, cls, 1, self.nzbins, figname='auto', titles=titles, ylabels=ylabels, **kwargs)


# @numba.jit(nopython=True, parallel=True)
def random_count(completeness, nobj):
    # even faster !
    return np.random.multinomial(nobj, pvals=completeness/np.sum(completeness)).astype(float)
