from .observable import *
from .utils import *
import sys
import numba
import random

TWOPI = 2. * np.pi

class Shear(Observable):
    def __init__(self, config, nside, mode, nzbins, mask_mode, data_dir='../data', dry_run=False, *args, **kwargs):
        self.obs_name = 'galaxy_shear'
        self.map_names = ['count', 'e1', 'e2']

        super(Shear, self).__init__(config, nside, mode, nzbins, self.obs_name, self.map_names, *args, **kwargs)

        self.spin = 2
        self.kernel = 'nz_source'
        self.type = twopoint.Types.galaxy_shear_emode_fourier

        self.data_dir = data_dir

        self.cats = {}

        mask_mode_split = mask_mode.split('>')
        assert mask_mode_split[0] in ['binary', 'count']
        self.mask_mode = mask_mode_split[0]
        if len(mask_mode_split) == 2:
            self.count_cut = int(mask_mode_split[1])
        else:
            self.count_cut = 0

        if not dry_run:
            if mode.startswith('buzzard'):
                self._init_buzzard()
            elif mode=='data_sub' or mode=='mastercat':
                self._init_data()
            elif mode=='mock':
                self._init_mock()
            elif mode=='full':
                self._init_full(kwargs['filename_template'], kwargs['dict'], kwargs['flip_e2'], kwargs.get('single_file', False), kwargs.get('ext_template', 'zbin_{}'), kwargs.get('ipix_instead_of_radec', False))
            elif mode=='tables':
                self._init_tables(kwargs['tables'], kwargs['dict'], kwargs['flip_e2'], kwargs.get('ext_template', 'zbin_{}', kwargs.get('ipix_instead_of_radec', False)))
            elif mode=='flask':
                self._init_flask(kwargs['isim'], kwargs['cookie'])
            elif mode=='psf' or 'dry':
                pass
            else:
                raise ValueError("Given `mode` argument ({}) is not correct.".format(mode))

    def _init_buzzard(self):
        # self.zlims = [(.2, .43), (.43,.63), (.64,.9), (.9, 1.3), (.2,1.3)][:self.nzbins]
        for ibin in self.prog(self.zbins):
            filename = os.path.join(self.data_dir, "Niall_WL_y3_bin_{}.fits".format(ibin+1))
            _cat = fits.open(filename)[1].data

            cat = {}
            cat['ra'] = _cat['RA']
            cat['dec'] = _cat['DEC']

            if self.mode == 'buzzard':
                cat['e1'] = _cat['E1']
                cat['e2'] = _cat['E2']
            if self.mode == 'buzzard_truth':
                cat['e1'] = _cat['G1']
                cat['e2'] = _cat['G2']

            self.cats[ibin] = cat

    def _init_data(self):
        for ibin in self.prog(self.zbins):
            filename = os.path.join(self.data_dir, "source_s{}.fits".format(ibin+1))
            _cat = fits.open(filename)[1].data

            cat = {}
            cat['ra'] = _cat['RA']
            cat['dec'] = _cat['DEC']

            cat['e1'] = _cat['g1']
            cat['e2'] = -1.0 * _cat['g2']

            self.cats[ibin] = cat

    def _init_mock(self):
        for ibin in self.prog(self.zbins):
            filename = os.path.join(self.data_dir, "source_s{}.fits".format(ibin+1))
            _cat = fits.open(filename)[1].data

            cat = {}
            cat['ra'] = _cat['RA']
            cat['dec'] = _cat['DEC']

            cat['e1'] = _cat['e1']
            cat['e2'] = _cat['e2']

            self.cats[ibin] = cat

    def _init_full(self, filename_template, dict, flip_e2, single_file=False, ext_template='zbin_{}', ipix_instead_of_radec=False):
        if single_file:
            full_cat = fits.open(os.path.join(self.data_dir, filename_template))
        if ipix_instead_of_radec:
            self.ipix = {}
        for ibin in self.prog(self.zbins):
            if single_file:
                _cat = full_cat[ext_template.format(ibin)].data
            else:
                filename = os.path.join(self.data_dir, filename_template.format(ibin+1))
                _cat = fits.open(filename)[1].data

            cat = {}
            if ipix_instead_of_radec:
                self.ipix[ibin] = _cat[dict['ipix']]
            else:
                cat['ra'] = _cat[dict['ra']]
                cat['dec'] = _cat[dict['dec']]

            cat['e1'] = _cat[dict['e1']]
            cat['e2'] = _cat[dict['e2']]

            if flip_e2:
                cat['e2'] *= -1.0

            self.cats[ibin] = cat    

    def _init_tables(self, tables, dict, flip_e2=False, ext_template='zbin_{}', ipix_instead_of_radec=False):
        is_list = isinstance(tables, list)
        if ipix_instead_of_radec:
            self.ipix = {}
        for i, ibin in self.prog(enumerate(self.zbins)):
            if is_list:
                _cat = tables[i]
            else:
                _cat = tables[ext_template.format(ibin)]

            cat = {}
            if ipix_instead_of_radec:
                self.ipix[ibin] = _cat[dict['ipix']]
            else:
                cat['ra'] = _cat[dict['ra']]
                cat['dec'] = _cat[dict['dec']]

            cat['e1'] = _cat[dict['e1']]
            cat['e2'] = _cat[dict['e2']]

            if flip_e2:
                cat['e2'] *= -1.0

            self.cats[ibin] = cat

    def _init_flask(self, isim, cookie):
        for ibin in self.prog(self.zbins):
            filename = os.path.join(self.data_dir, 'src-cat_s{}_z{}_ck{}.fits'.format(isim, ibin+1, cookie))
            _cat = fits.open(filename)[1].data

            cat = {}
            cat['ra'] = _cat['RA']
            cat['dec'] = _cat['DEC']

            cat['e1'] = _cat['GAMMA1']
            cat['e2'] = -1.0 * _cat['GAMMA2']

            self.cats[ibin] = cat   

    def make_maps(self, save=True):
        keys = ['e1', 'e2']
        self.get_ipix()
        for ibin in self.prog(self.zbins, desc='{}.make_maps'.format(self.obs_name)):
            cat = self.cats[ibin]
            quantities, count, mask = ca.cosmo.make_healpix_map(None, None,
                                                    quantity=[cat[_x] for _x in keys],
                                                    nside=self.nside, fill_UNSEEN=True,
                                                    mask=None, weight=None,
                                                    ipix=self.ipix[ibin])
            for j, key in enumerate(keys):
                self.maps[ibin][key] = quantities[j]

            # Count cut
            count_cut_mask = (count>self.count_cut)

            self.maps[ibin]['count'] = count * count_cut_mask.astype(int)
            self.masks[ibin] = mask.astype(float) * count_cut_mask.astype(float)

        if save:
            self.save_maps()        

    def make_masks_apo(self):
        super(Shear, self).make_masks_apo()

        # Apply inverse-variance weighting after apodization
        if self.mask_mode =='count':
            for ibin in self.zbins:
                self.masks_apo[ibin] *= self.maps[ibin]['count']

    def make_fields(self, hm, include_templates=True):
        if hm.purify_b or hm.purify_e:
            print("WARNING: E/B-mode purification requires unseen pixels to be set to zero. Replacing...")
            for ibin in self.zbins:
                self.maps[ibin]['e1'] = hpunseen2zero(self.maps[ibin]['e1'])
                self.maps[ibin]['e2'] = hpunseen2zero(self.maps[ibin]['e2'])
        
        templates = self._get_templates_array() if include_templates else None

        for ibin in self.prog(self.zbins, desc='{}.make_fields'.format(self.obs_name)):
            self.fields[ibin] = nmt.NmtField(self.masks_apo[ibin],
                                             [self.maps[ibin]['e1'], self.maps[ibin]['e2']],
                                             templates=templates,
                                             **hm.fields_kw)

    def make_randomized_fields(self, hm, ibin, nrandom=1, include_templates=True):
        bool_mask = (self.maps[ibin]['count'] > 0.)
        self.get_ipix()
        fields = []

        templates = self._get_templates_array() if include_templates else None

        # remove progress bar for only one field
        if nrandom == 1:
            prog = [0]
        else:
            prog = self.prog(nrandom)

        for _ in prog:
            e1_map, e2_map = _randrot_maps(self.cats[ibin]['e1'].astype(float), self.cats[ibin]['e2'].astype(float), self.ipix[ibin], self.npix, bool_mask, self.maps[ibin]['count'])
            field =  nmt.NmtField(self.masks_apo[ibin],
                                  [e1_map, e2_map],
                                  templates=templates,
                                  **hm.fields_kw)
            fields.append(field)

        return fields

    def _get_info(self):
        import pandas as pd

        info = {}
        info['e1std'] = []
        info['e2std'] = []
        info['e12std'] = []
        info['fsky'] = []
        info['Ngal'] = []
        info['area (std)'] = []
        info['area (sq. deg.)'] = []
        info['nbar (gal/std)'] = []
        info['nbar (gal/arcmin)'] = []

        for i in self.zbins:
            e1std = np.std(self.cats[i]['e1'])
            e2std = np.std(self.cats[i]['e2'])
            e12std = np.sqrt(e1std**2+e2std**2)
            fsky = np.sum(self.maps[i]['count']>0.) * 1./ len(self.masks[i])
            Ngal = int(np.sum(self.maps[i]['count'][self.masks[i].astype(bool)]))
            area = 4.*np.pi * fsky
            area_sqdeg = area * (180./np.pi)**2
            nbar_std = Ngal / area
            nbar_arcmin = nbar_std / (180*60/np.pi)**2
            
            info['e1std'].append(e1std)
            info['e2std'].append(e2std)
            info['e12std'].append(e12std)
            info['fsky'].append(fsky)
            info['Ngal'].append(Ngal)
            info['area (std)'].append(area)
            info['area (sq. deg.)'].append(area_sqdeg)
            info['nbar (gal/std)'].append(nbar_std)
            info['nbar (gal/arcmin)'].append(nbar_arcmin)

        df = pd.DataFrame(index=self.zbins, data=info)
        return df

    def compute_ipix(self): 
        self.ipix = {}
        for ibin in self.prog(self.zbins, desc='{}.compute_ipix'.format(self.obs_name)):
            cat = self.cats[ibin]
            self.ipix[ibin] = hp.ang2pix(self.nside, (90-cat['dec'])*np.pi/180.0, cat['ra']*np.pi/180.0)
    
    def set_ipix(self, ipixs):
        self.ipix = {}
        for ibin, ipix in zip(self.zbins, ipixs):
            self.ipix[ibin] = ipix
    
    def get_ipix(self):
        if not hasattr(self, 'ipix'):
            self.compute_ipix()
        return self.ipix

    # @profile
    def _compute_random_auto_cls(self, hm, ibin, nrandom):
        npix = self.npix
        cat = self.cats[ibin]
        mask_apo = self.masks_apo[ibin]

        wsp = hm.get_workspace(self, self, ibin, ibin)#, save_wsp=save_wsp)
        wsp_filename = hm.get_workspace_filename(self, self, ibin, ibin)

        # Nobj = len(cat)
        self.get_ipix()

        ipix = self.ipix[ibin] #hp.ang2pix(self.nside, (90-cat['dec'])*np.pi/180.0, cat['ra']*np.pi/180.0)
        count = self.maps[ibin]['count']
        bool_mask = (count > 0.)

        _cls = []

        if hm.nproc==0:
            for _ in self.prog(nrandom, desc='{}._compute_random_auto_cls [bin {}]'.format(self.obs_name, ibin)):
                _cls.append(_randrot_cls(cat['e1'].astype(float), cat['e2'].astype(float), ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, wsp))

        else:
            args = (cat['e1'], cat['e2'], ipix, npix, bool_mask, mask_apo, count, hm.purify_e, hm.purify_b, wsp_filename) # self.nside, hm.lmax, hm.nlb)
            _multiple_results = [hm.pool.apply_async(_multiproc_randrot_cls, (len(_x), args, pos+1)) for pos, _x in enumerate(np.array_split(range(nrandom), hm.nproc)) if len(_x)>0]
            for res in self.prog(_multiple_results, desc='{}._compute_random_auto_cls [bin {}]<{}>'.format(self.obs_name, ibin, os.getpid()), position=0):
                _cls += res.get()
            print("\n")

        return np.array(_cls)

    def plot_auto_cls(self, hm, remove_Nl=False, **kwargs):
        cls = {}
        titles = {}
        ylabels = {}

        _titles = ['EE', 'EB', 'BB']

        idx_EB = [0, 1, 3]

        for i, zbin in enumerate(self.zbins):
            for j in range(3):
                k = (i,j)
                titles[k] = _titles[j] + ' [bin %i]'%(i+1)
                ylabels[k] = '$C_\\ell$'
                ylabels[k] = '$C_\\ell$'
                if 'factor_ell' in kwargs.keys():
                    if kwargs['factor_ell'] == 1:
                        ylabels[k] = '$\\ell C_\\ell$'
                    if kwargs['factor_ell'] == 2:
                        ylabels[k] = '$\\ell (\\ell+1) C_\\ell$'
                else:
                    ylabels[k] = '$C_\\ell$'
                cls[k] = {}
                cls[k]['data'] = np.copy(hm.cls[(self.obs_name, self.obs_name)][(zbin, zbin)]['data'][idx_EB[j]])
                cls[k]['random'] = np.copy(hm.cls[(self.obs_name, self.obs_name)][(zbin, zbin)]['random'][:,idx_EB[j],:])
                if remove_Nl:
                    clr_r_m = np.mean(cls[k]['random'], axis=0)
                    cls[k]['data'] -= clr_r_m
                    cls[k]['random'] -= clr_r_m

        return self.plot_cls(hm, cls, self.nzbins, 3, figname='auto', titles=titles, ylabels=ylabels, **kwargs)

    def plot_cls_BB_only(self, hm, remove_Nl=False, **kwargs):
        cls = {}
        titles = {}
        ylabels = {}

        for i in range(1):
            for j, zbin in enumerate(self.zbins):
                k = (i,j)
                titles[k] = 'BB spectrum [bin %i]'%(zbin+1)
                if 'factor_ell' in kwargs.keys():
                    if kwargs['factor_ell'] == 1:
                        ylabels[k] = '$\\ell C_\\ell ^{\\rm BB}$'
                    if kwargs['factor_ell'] == 2:
                        ylabels[k] = '$\\ell (\\ell+1) C_\\ell ^{\\rm BB}$'
                else:
                    ylabels[k] = '$C_\\ell ^{\\rm BB}$'
                cls[k] = {}
                cls[k]['data'] = np.copy(hm.cls[(self.obs_name, self.obs_name)][(zbin, zbin)]['data'][3])
                cls[k]['random'] = np.copy(hm.cls[(self.obs_name, self.obs_name)][(zbin, zbin)]['random'][:,3,:])
                if remove_Nl:
                    clr_r_m = np.mean(cls[k]['random'], axis=0)
                    cls[k]['data'] -= clr_r_m
                    cls[k]['random'] -= clr_r_m

        return self.plot_cls(hm, cls, 1, self.nzbins, figname='BB', titles=titles, ylabels=ylabels, **kwargs)


@numba.jit(nopython=True, parallel=True)
def _randrot_maps_sub(cat_e1, cat_e2, ipix, npix):
    n = len(cat_e1)
    e1_map = np.zeros(npix)
    e2_map = np.zeros(npix)
    for i in numba.prange(n):
        rot_angle = random.random() * TWOPI
        cos = np.cos(rot_angle)
        sin = np.sin(rot_angle)
        e1_map[ipix[i]] += + cat_e1[i] * cos + cat_e2[i] * sin
        e2_map[ipix[i]] += - cat_e1[i] * sin + cat_e2[i] * cos
    
    return e1_map, e2_map

def _randrot_maps(cat_e1, cat_e2, ipix, npix, bool_mask, count):
    e1_map, e2_map = _randrot_maps_sub(cat_e1, cat_e2, ipix, npix)

    e1_map[bool_mask] /= count[bool_mask]
    e2_map[bool_mask] /= count[bool_mask]

    return e1_map, e2_map

def _randrot_field(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b):
    e1_map, e2_map = _randrot_maps(cat_e1, cat_e2, ipix, npix, bool_mask, count)
    return nmt.NmtField(mask_apo, [e1_map, e2_map], purify_e=purify_e, purify_b=purify_b)

def _randrot_cls(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, wsp):
    field = _randrot_field(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b)
    cls = compute_master(field, field, wsp)
    return cls


def _multiproc_randrot_cls(nsamples, args, pos):
    cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, wsp_filename = args

    wsp = nmt.NmtWorkspace()
    wsp.read_from(wsp_filename)

    _cls = []
    for _ in trange(nsamples, desc="[worker {:4d}]<{}>".format(pos,os.getpid()), position=pos, leave=False):
        _cls.append(_randrot_cls(cat_e1, cat_e2, ipix, npix, bool_mask, mask_apo, count, purify_e, purify_b, wsp))

    return _cls


