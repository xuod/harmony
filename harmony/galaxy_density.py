from .observable import *
import numba


class Galaxy(Observable):
    def __init__(self, config, nside, mode, nzbins, data_dir='../data', mask_dir='../masks', use_weights=False, dry_run=False, true_density=True, *args, **kwargs):

        self.obs_name = 'galaxy_density'
        self.map_names = ['count', 'density', 'completeness']

        super(Galaxy, self).__init__(config, nside, mode, nzbins, self.obs_name, self.map_names, *args, **kwargs)

        self.spin = 0
        self.kernel = 'nz_lens'
        self.type = twopoint.Types.galaxy_position_fourier

        self.data_dir = data_dir
        self.mask_dir = mask_dir

        self.true_density = true_density

        self.cats = {}

        self.has_weights = False

        if not dry_run:
            if mode=='redmagic_Y3':
                self.init_redmagic_Y3(self.nzbins)
            elif mode=='maglim_Y3':
                self.init_maglim_Y3(self.nzbins)
            elif mode=='redmagic_Y1':
                self.init_redmagic_Y1(self.nzbins)
            elif mode=='BAO_Y1':
                self.init_BAO_Y1(self.nzbins, use_weights)
            else:
                raise NotImplementedError

    def init_redmagic_Y3(self, nzbins):
        # basename = 'y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_'
        # cats_name = ['redmagic_highdens_0.5', 'redmagic_highlum_1.0', 'redmagic_higherlum_1.5'] #
        # mask_ext = '_vlim_zmask.fit'
        # cats_ext = ['-10.fit', '-04.fit', '-01.fit']

        # which_cat_zbins = [0,0,0,1,2]

        for ibin in self.prog(self.zbins, desc='Galaxy.init_redmagic_Y3'):
            self.cats[ibin] = fits.open(os.path.join(self.data_dir, 'redmagic_bin{}.fits'.format(ibin)))[1].data
            # self.masks[ibin] = hp.read_map(os.path.join(self.mask_dir, basename+cats_name[which_cat_zbins[ibin]]+'_binary_nside%i.fits'%(self.nside)), verbose=False)
            # comp = hp.read_map(os.path.join(self.mask_dir, basename+cats_name[which_cat_zbins[ibin]]+'_FRACGOOD_nside%i.fits'%(self.nside)), verbose=False)
            # comp[comp == hp.UNSEEN] = 0.0
            # self.maps[ibin]['completeness'] = comp
            self.masks[ibin] = hpunseen2zero(hp.read_map(os.path.join(self.data_dir, 'redmagic_bin{}_binary_nside{}.fits'.format(ibin, self.nside)), verbose=False))
            self.maps[ibin]['completeness'] = hpunseen2zero(hp.read_map(os.path.join(self.data_dir, 'redmagic_bin{}_comp_nside{}.fits'.format(ibin, self.nside)), verbose=False))

    def init_redmagic_Y1(self, nzbins):
        basename = '5bins_hidens_hilum_higherlum_jointmask_0.15-0.9_magauto_mof_combo_removedupes_spt_fwhmi_exptimei_cut_badpix_mask'

        for ibin in self.prog(self.zbins, desc='Galaxy.init_redmagic_Y1'):
            self.cats[ibin] = fits.open(os.path.join(self.data_dir, 'redmagic_bin{}.fits'.format(ibin)))[1].data
            self.masks[ibin] = hp.read_map(os.path.join(self.mask_dir, basename+'_binary_nside{}.fits'.format(self.nside)), verbose=False)
            comp = hp.read_map(os.path.join(self.mask_dir, basename+'_FRACGOOD_nside{}.fits'.format(self.nside)), verbose=False)
            comp[comp == hp.UNSEEN] = 0.0
            self.maps[ibin]['completeness'] = comp
    
    def init_BAO_Y1(self, nzbins, use_weights):
        basename = 'DES_Y1A1_LSSBAO_v1.0_MASK_HPIX4096RING'

        for ibin in self.prog(self.zbins, desc='Galaxy.init_BAO_Y1'):
            self.cats[ibin] = fits.open(os.path.join(self.data_dir, 'redmagic_bin{}.fits'.format(ibin)))[1].data
            self.masks[ibin] = hp.read_map(os.path.join(self.mask_dir, basename+'_binary_nside{}.fits'.format(self.nside)), verbose=False)
            comp = hp.read_map(os.path.join(self.mask_dir, basename+'_FRAC_nside{}.fits'.format(self.nside)), verbose=False)
            comp[comp == hp.UNSEEN] = 0.0
            self.maps[ibin]['completeness'] = comp
        self.has_weights = use_weights

    def init_maglim_Y3(self, nzbins):
        for ibin in self.prog(self.zbins, desc='Galaxy.init_maglim_Y3'):
            self.cats[ibin] = fits.open(os.path.join(self.data_dir, 'maglim_bin{}.fits'.format(ibin+1)))[1].data
            self.masks[ibin] = hp.read_map(os.path.join(self.data_dir, 'maglim_bin{}_binary_nside{}.fits'.format(ibin+1, self.nside)), verbose=False)
            comp = hp.read_map(os.path.join(self.data_dir, 'maglim_bin{}_comp_nside{}.fits'.format(ibin+1, self.nside)), verbose=False)
            comp[comp == hp.UNSEEN] = 0.0
            self.maps[ibin]['completeness'] = comp

    def make_maps(self, save=True):
        for ibin in self.prog(self.zbins, desc='Galaxy.make_maps'):
            cat = self.cats[ibin]
            if self.has_weights:
                count_w, count, _ = ca.cosmo.make_healpix_map(cat['ra'], cat['dec'],
                                            [np.ones_like(cat['dec'])], self.nside,
                                            mask=None,
                                            weight=[cat['weight']],
                                            fill_UNSEEN=False, return_extra=False, mode='sum')

                density = ca.cosmo.count2density(count_w[0], mask=self.masks[ibin], completeness=self.maps[ibin]['completeness'], true_density=self.true_density)

            else:
                _, count, _ = ca.cosmo.make_healpix_map(cat['ra'], cat['dec'], None, self.nside,
                                            mask=self.masks[ibin],
                                            weight=None,
                                            fill_UNSEEN=False, return_extra=False)

                density = ca.cosmo.count2density(count, mask=self.masks[ibin], completeness=self.maps[ibin]['completeness'], true_density=self.true_density)

            self.maps[ibin]['count'] = count
            self.maps[ibin]['density'] = density

        if save:
            self.save_maps()

    # def make_fields(self, hm, include_templates=True):
    #     for ibin in self.prog(self.zbins, desc='{}.make_fields'.format(self.obs_name)):
    #         self.fields[ibin] = nmt.NmtField(self.masks_apo[ibin], [self.maps[ibin]['density']], templates=self._get_templates_array(), purify_e=hm.purify_e, purify_b=hm.purify_b)

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
            fsky = np.sum(self.masks[i]>0.) * 1./ len(self.masks[i])
            fsky_eff = np.sum(self.maps[i]['completeness']) * 1./ len(self.maps[i]['completeness'])
            Ngal = int(np.sum(self.maps[i]['count'][self.masks[i].astype(bool)]))
            area = 4.*np.pi * fsky
            area_eff = 4.*np.pi * fsky_eff
            nbar = Ngal / area
            nbar_eff = Ngal / area_eff
            
            info['fsky'].append(fsky)
            info['fsky (effective)'].append(fsky_eff)
            info['Ngal'].append(Ngal)
            info['area (std)'].append(area)
            info['area (effective, std)'].append(area_eff)
            info['nbar (std)'].append(nbar)
            info['nbar (effective, std)'].append(nbar_eff)

        df = pd.DataFrame(index=self.zbins, data=info)
        return df
    
    def make_randomized_maps(self, ibin):
        # npix = hp.nside2npix(self.nside)
        # mask_apo = self.masks_apo[ibin]

        Nobj = len(self.cats[ibin]['ra'])

        # completeness = self.maps[ibin]['completeness']
        # print(completeness.min(), completeness.max(), np.all(np.isfinite(completeness)))
        count = random_count(self.maps[ibin]['completeness'], Nobj)
        # import pdb; pdb.set_trace()
        density = ca.cosmo.count2density(count, mask=self.masks[ibin], completeness=self.maps[ibin]['completeness'])

        return [density]

    # def _compute_random_auto_cls(self, hm, ibin, nrandom, use_completeness=False):
    #     npix = hp.nside2npix(self.nside)

    #     mask_apo = self.masks_apo[ibin]

    #     wsp = hm.get_workspace(self, self, ibin, ibin)#, save_wsp=save_wsp) #wsp = nmt.NmtWorkspace()

    #     Nobj = len(self.cats[ibin])

    #     _cls = []

    #     if use_completeness:
    #         random_comp = self.masks[ibin].astype(float)
    #     else:
    #         random_comp = self.maps[ibin]['completeness']

    #     for i in self.prog(nrandom, desc='Galaxy.compute_cls [bin {}]'.format(ibin)):
    #         count = random_count(random_comp, Nobj)
    #         density = ca.cosmo.count2density(count, mask=self.masks[ibin], completeness=random_comp)
    #         field_r = nmt.NmtField(mask_apo, [density], templates=None, purify_e=hm.purify_e, purify_b=hm.purify_b)
    #         _cls.append(compute_master(field_r, field_r, wsp))

    #     return np.array(_cls)

    def plot_auto_cls(self, hm, remove_Nl=False, **kwargs):
        cls = {}
        titles = {}
        ylabels = {}

        idx_EB = [0, 1, 3]

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


# def random_pos(completeness, nobj):
#     return np.random.choice(len(completeness), size=nobj, replace=True, p=completeness*1./np.sum(completeness))

# @numba.jit(nopython=True, parallel=True)
def random_count(completeness, nobj):
    # even faster !
    return np.random.multinomial(nobj, pvals=completeness/np.sum(completeness)).astype(float)
