from .observable import *

class Galaxy(Observable):
    def __init__(self, config, nside, mode, nzbins, data_dir='../data', mask_dir='../masks', use_weights=False, *args, **kwargs):

        self.obs_name = 'galaxy_density'
        self.map_names = ['count', 'density', 'completeness']

        super(Galaxy, self).__init__(config, nside, mode, nzbins, self.obs_name, self.map_names, *args, **kwargs)

        self.data_dir = data_dir
        self.mask_dir = mask_dir

        self.cats = {}

        self.has_weights = False

        if mode=='redmagic_Y3':
            self.init_redmagic_Y3(self.nzbins)

        if mode=='redmagic_Y1':
            self.init_redmagic_Y1(self.nzbins)
        
        if mode=='BAO_Y1':
            self.init_BAO_Y1(self.nzbins, use_weights)

    def init_redmagic_Y3(self, nzbins):
        basename = 'y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_'
        cats_name = ['redmagic_highdens_0.5', 'redmagic_highlum_1.0', 'redmagic_higherlum_1.5'] #
        mask_ext = '_vlim_zmask.fit'
        cats_ext = ['-10.fit', '-04.fit', '-01.fit']

        which_cat_zbins = [0,0,0,1,2]

        for ibin in tqdm(self.zbins, desc='Galaxy.init_redmagic_Y3'):
            self.cats[ibin] = fits.open(os.path.join(self.data_dir, 'redmagic_bin{}.fits'.format(ibin)))[1].data
            self.masks[ibin] = hp.read_map(os.path.join(self.mask_dir, basename+cats_name[which_cat_zbins[ibin]]+'_binary_nside%i.fits'%(self.nside)), verbose=False)
            comp = hp.read_map(os.path.join(self.mask_dir, basename+cats_name[which_cat_zbins[ibin]]+'_FRACGOOD_nside%i.fits'%(self.nside)), verbose=False)
            comp[comp == hp.UNSEEN] = 0.0
            self.maps[ibin]['completeness'] = comp


    def init_redmagic_Y1(self, nzbins):
        basename = '5bins_hidens_hilum_higherlum_jointmask_0.15-0.9_magauto_mof_combo_removedupes_spt_fwhmi_exptimei_cut_badpix_mask'

        for ibin in tqdm(self.zbins, desc='Galaxy.init_redmagic_Y1'):
            self.cats[ibin] = fits.open(os.path.join(self.data_dir, 'redmagic_bin{}.fits'.format(ibin)))[1].data
            self.masks[ibin] = hp.read_map(os.path.join(self.mask_dir, basename+'_binary_nside{}.fits'.format(self.nside)), verbose=False)
            comp = hp.read_map(os.path.join(self.mask_dir, basename+'_FRACGOOD_nside{}.fits'.format(self.nside)), verbose=False)
            comp[comp == hp.UNSEEN] = 0.0
            self.maps[ibin]['completeness'] = comp
    
    def init_BAO_Y1(self, nzbins, use_weights):
        basename = 'DES_Y1A1_LSSBAO_v1.0_MASK_HPIX4096RING'

        for ibin in tqdm(self.zbins, desc='Galaxy.init_BAO_Y1'):
            self.cats[ibin] = fits.open(os.path.join(self.data_dir, 'redmagic_bin{}.fits'.format(ibin)))[1].data
            self.masks[ibin] = hp.read_map(os.path.join(self.mask_dir, basename+'_binary_nside{}.fits'.format(self.nside)), verbose=False)
            comp = hp.read_map(os.path.join(self.mask_dir, basename+'_FRAC_nside{}.fits'.format(self.nside)), verbose=False)
            comp[comp == hp.UNSEEN] = 0.0
            self.maps[ibin]['completeness'] = comp
            self.has_weights = use_weights

    def make_maps(self, save=True):
        for ibin in tqdm(self.zbins, desc='Galaxy.make_maps'):
            cat = self.cats[ibin]
            if self.has_weights:
                count_w, count, _ = ca.cosmo.make_healpix_map(cat['ra'], cat['dec'], [np.ones_like(cat['dec'])], self.nside,
                                            mask=None,
                                            weight=[cat['weight']],
                                            fill_UNSEEN=False, return_extra=False, mode='sum')

                density = ca.cosmo.count2density(count_w[0], mask=self.masks[ibin], completeness=self.maps[ibin]['completeness'])

            else:
                _, count, _ = ca.cosmo.make_healpix_map(cat['ra'], cat['dec'], None, self.nside,
                                            mask=self.masks[ibin],
                                            weight=cat['weight'] if self.has_weights else None,
                                            fill_UNSEEN=False, return_extra=False)

                density = ca.cosmo.count2density(count, mask=self.masks[ibin], completeness=self.maps[ibin]['completeness'])

            self.maps[ibin]['count'] = count
            self.maps[ibin]['density'] = density

        if save:
            self.save_maps()

    def get_field(self, hm, ibin, include_templates=True):
        return nmt.NmtField(self.masks_apo[ibin], [self.maps[ibin]['density']], templates=self._get_templates_array(), purify_e=hm.purify_e, purify_b=hm.purify_b)

    def _compute_auto_cls(self, hm, ibin, nrandom=0, save=True):
        npix = hp.nside2npix(self.nside)

        mask_apo = self.masks_apo[ibin]

        wsp = nmt.NmtWorkspace()
        field_0 = self.get_field(hm, ibin) #nmt.NmtField(mask_apo, [self.maps[ibin]['density']], templates=self.templates, purify_e=hm.purify_e, purify_b=hm.purify_b)

        wsp.compute_coupling_matrix(field_0, field_0, hm.b)

        cls = {}
        cls['true'] = compute_master(field_0, field_0, wsp)

        if nrandom > 0:
            Nobj = len(self.cats[ibin])

            _cls = []

            for i in trange(nrandom, desc='Galaxy.compute_cls [bin {}]'.format(ibin)):
                count = np.zeros(npix, dtype=float)
                ipix_r = random_pos(self.masks[ibin].astype(float), Nobj)
                np.add.at(count, ipix_r, 1.)

                density = ca.cosmo.count2density(count, mask=self.masks[ibin], completeness=None)

                field_r = nmt.NmtField(mask_apo, [density], templates=None, purify_e=hm.purify_e, purify_b=hm.purify_b)

                _cls.append(compute_master(field_r, field_r, wsp))

            cls['random'] = np.array(_cls)

        return cls

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
            cls[k]['true'] = np.copy(hm.cls[(self.obs_name, self.obs_name)][zbin]['true'][0])
            cls_r = hm.cls[(self.obs_name, self.obs_name)][zbin]['random'][:,0,:]
            if remove_Nl:
                cls[k]['true'] -= np.mean(cls_r, axis=0)
            else:
                cls[k]['random'] = cls_r

        return self.plot_cls(hm, cls, 1, self.nzbins, figname='auto', titles=titles, ylabels=ylabels, **kwargs)


def random_pos(completeness, nobj):
    return np.random.choice(len(completeness), size=nobj, replace=True, p=completeness*1./np.sum(completeness))
