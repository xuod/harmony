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
import numpy as np
import logging
import scipy
import twopoint

class Observable(object):
    def __init__(self, config, nside, mode, nzbins, obs_name, map_names, suffix='', verbose=True, aposize=0.0, apotype='C1', **fields_kw):
        self.config = config
        self.name = config.name
        self.nside = nside
        self.npix = hp.nside2npix(nside)

        # Apodization
        self.aposize = aposize
        self.apotype = apotype

        # Field options
        self.fields_kw = fields_kw # {}
        # self.fields_kw['templates'] = None
        # self.fields_kw['beam'] = kwargs.get('beam', None)
        # self.fields_kw['purify_e'] = kwargs.get('purify_e', False)
        # self.fields_kw['purify_b'] = kwargs.get('purify_b', False)
        # self.fields_kw['n_iter_mask_purify'] = kwargs.get('n_iter_mask_purify', 3)
        # self.fields_kw['tol_pinv'] = kwargs.get('tol_pinv', 1e-10)
        # self.fields_kw['wcs'] = kwargs.get('wcs', None)
        # self.fields_kw['n_iter'] = kwargs.get('n_iter', 3)
        # self.fields_kw['lmax_sht'] = kwargs.get('lmax_sht', -1)

        self.map_names = map_names
        self.obs_name = obs_name
        if suffix != '':
            self.obs_name += '_' + suffix
        self.mode = mode
        # self.nzbins = nzbins
        self.spin = None

        if type(nzbins) == int:
            self.zbins = list(range(nzbins))
            self.nzbins = nzbins
        else:
            self.zbins = nzbins
            self.nzbins = len(nzbins)

        self.maps = {} # to be organized as maps[redshift_bin][map_name]
        self.fields = {} # to be organized as maps[redshift_bin][map_name]
        self.masks = {} # to be organized as maps[redshift_bin][map_name] # NO !
        self.masks_apo = {} # to be organized as maps[redshift_bin][map_name] # NO !

        for i in self.zbins:
            self.masks[i] = None
            self.masks_apo[i] = None
            self.maps[i] = {}
            self.fields[i] = None
            for name in map_names:
                self.maps[i][name] = None

        # self.nproc = nproc
        # if nproc > 1:
        #     self.pool = multiprocessing.Pool(nproc)

        self.has_templates = False

        self.verbose = verbose
        self.prog = prog(verbose)

    # def load_catalogs(self):
    #     print('Method load_catalogs not implemented, nothing to do.')

    def clear(self):
        self.maps = {} # to be organized as maps[redshift_bin][map_name]
        self.fields = {} # to be organized as maps[redshift_bin][map_name]
        self.masks = {} # to be organized as maps[redshift_bin][map_name] # NO !
        self.masks_apo = {} # to be organized as maps[redshift_bin][map_name] # NO !

        for i in self.zbins:
            self.masks[i] = None
            self.masks_apo[i] = None
            self.maps[i] = {}
            self.fields[i] = None
            for name in self.map_names:
                self.maps[i][name] = None

    def make_maps(self):
        raise NotImplementedError

    def save_maps(self):
        maps_dir = os.path.join(self.config.path_maps, self.name)
        make_directory(maps_dir)
        for ibin in self.prog(self.zbins, desc='{}.save_maps'.format(self.obs_name)):
            hp.write_map(os.path.join(maps_dir, '{}_{}_{}_{}_nside{}_bin{}.fits'.format(self.obs_name, 'mask', self.config.name, self.mode, self.nside, ibin)), self.masks[ibin], overwrite=True)
            for map_name in self.map_names:
                hp.write_map(os.path.join(maps_dir, '{}_{}_{}_{}_nside{}_bin{}.fits'.format(self.obs_name, map_name, self.config.name, self.mode, self.nside, ibin)), self.maps[ibin][map_name], overwrite=True)

    def load_maps(self):
        maps_dir = os.path.join(self.config.path_maps, self.name)
        for ibin in self.prog(self.zbins, desc='{}.load_maps'.format(self.obs_name)):
            # self.maps[ibin] = {}
            self.masks[ibin] = hp.read_map(os.path.join(maps_dir, '{}_{}_{}_{}_nside{}_bin{}.fits'.format(self.obs_name, 'mask', self.config.name, self.mode, self.nside, ibin)), verbose=False)
            for map_name in self.map_names:
                self.maps[ibin][map_name] = hp.read_map(os.path.join(maps_dir, '{}_{}_{}_{}_nside{}_bin{}.fits'.format(self.obs_name, map_name, self.config.name, self.mode, self.nside, ibin)), verbose=False)

    def plot_maps(self, subplots=True, show_masks=True, **kwargs):
        make_directory(self.config.path_figures+'/'+self.name)
        if subplots:
            nrow = self.nzbins
            ncol = len(self.map_names)
            if show_masks:
                ncol += 1 
            fig, axes = plt.subplots(nrow,ncol, figsize=(4*ncol,3*nrow))
            axes = np.atleast_2d(axes)

        for i, ibin in self.prog(enumerate(self.zbins), desc='{}.plot_maps'.format(self.obs_name)):
            for j, map_name in enumerate(self.map_names):
                if subplots:
                    plt.sca(axes[i,j])
                    hp.mollview(self.maps[ibin][map_name], title='{} (bin {})'.format(map_name, ibin), hold=True, **kwargs)
                else:
                    hp.mollview(self.maps[ibin][map_name], title='{} (bin {})'.format(map_name, ibin), **kwargs)
                    figfile = os.path.join(self.config.path_figures, self.name, '{}_{}_{}_nside{}_bin{}.png'.format(map_name, self.config.name, self.mode, self.nside, ibin))
                    plt.savefig(figfile, dpi=300)
                    plt.show()
            if show_masks:
                if subplots:
                    plt.sca(axes[i,j+1])
                    hp.mollview(self.masks[ibin], title='{} (bin {})'.format('mask', ibin), hold=True, **kwargs)
                else:
                    hp.mollview(self.masks[ibin], title='{} (bin {})'.format('mask', ibin), **kwargs)
                    figfile = os.path.join(self.config.path_figures, self.name, '{}_{}_{}_nside{}_bin{}.png'.format('mask', self.config.name, self.mode, self.nside, ibin))
                    plt.savefig(figfile, dpi=300)
                    plt.show()

        
        if subplots:
            # plt.tight_layout()
            figfile = os.path.join(self.config.path_figures, self.name, 'maps_{}_{}_nside{}_bin{}.png'.format(self.config.name, self.mode, self.nside, ibin))
            plt.savefig(figfile, dpi=300)
            plt.show()
    
    def make_masks_apo(self):
        self.masks_apo = {}
        for ibin in self.prog(self.zbins, desc='{}.make_masks_apo'.format(self.obs_name)):
            self.masks_apo[ibin] = nmt.mask_apodization(self.masks[ibin], aposize=self.aposize, apotype=self.apotype)
    
    # def make_fields(self, hm, include_templates=True):
    #     raise NotImplementedError

    def prepare_fields(self):
        raise NotImplementedError

    def make_fields(self, include_templates=True):
        maps = self.prepare_fields() 

        if self.fields_kw.get('purify_b', False) or self.fields_kw.get('purify_e', False):
            print("WARNING: E/B-mode purification requires unseen pixels to be set to zero. Replacing...")
            for ibin in self.zbins:
                maps[ibin] = [hpunseen2zero(m) for m in maps[ibin]]

        if include_templates:
            templates_fields_kw = self.fields_kw.get('templates', None)
            templates_arr = self._get_templates_array()
            assert (templates_arr is None) or (templates_fields_kw is None), "[make_fields] conflict for templates: defined both as kwarg and manually."
            if templates_fields_kw is None:
                self.fields_kw['templates'] = templates_arr

        for ibin in self.prog(self.zbins, desc='{}.make_fields'.format(self.obs_name)):
            self.fields[ibin] = nmt.NmtField(self.masks_apo[ibin], maps[ibin], **self.fields_kw)

    def get_field(self, ibin):
        if self.fields is not None:
            if ibin in self.fields:
                return self.fields[ibin]
            else:
                raise KeyError("Specified bin is not in field")
        else:
            raise RuntimeError("Fields are not constructed yet, use make_fields first.")

    # def make_randomized_fields(self, hm, ibin, nrandom=1):
    #     raise NotImplementedError

    def make_randomized_maps(self, ibin):
        raise NotImplementedError

    def make_randomized_fields(self, ibin, nrandom=1):
        # remove progress bar for only one field
        if nrandom == 1:
            prog = [0]
        else:
            prog = self.prog(nrandom)

        fields = []
        for _ in prog:
            random_maps = self.make_randomized_maps(ibin)
            field = nmt.NmtField(self.masks_apo[ibin], random_maps, **{k:self.fields_kw[k] for k in self.fields_kw.keys() if k!='templates'})
            fields.append(field)

        return fields

    def _get_info(self):
        raise NotImplementedError

    def get_info(self, tofile=None):
        df = self._get_info()

        if tofile is not None:
            df.to_csv(tofile)

        return df

    def _get_templates_array(self):
        if not self.has_templates:
            return None
        else:
            templates = []
            for key, temp in self.templates_dir.items():
                templates.append(list(temp))
            # templates = np.expand_dims(np.array(templates), axis=1)
            return np.array(templates)

    def _check_init_templates(self):
        if not hasattr(self, 'templates_dir'):
            self.has_templates = True
            self.templates_dir = {}

    def add_template(self, template_name, template_map):
        self._check_init_templates()
        self.templates_dir[template_name] = template_map

    def load_template(self, template_name, template_filename):
        self._check_init_templates()
        self.templates_dir[template_name] =  hp.read_map(template_filename, verbose=False)

    def load_all_templates_from_dir(self, templates_dir):
        self._check_init_templates()

        template_names = os.listdir(templates_dir)
        template_names.sort()
        for filename in self.prog(template_names, desc='{}.load_all_templates_from_dir'.format(self.obs_name)):
            # try:
                temp = hp.read_map(os.path.join(templates_dir, filename), verbose=False)
                self.templates_dir[filename] = temp
            # except:
            #     print("Could not read {}".format(os.path.join(templates_dir, filename)))
            # self.templates.append(temp)

        # self.templates = np.array(self.templates)
        # self.templates = np.expand_dims(self.templates, axis=1)

    def load_DES_templates(self, templates_dir, bands=['g', 'r', 'i', 'z']):
        self._check_init_templates()

        self.bands = bands
        self.syst = {}
        self.syst['Air mass'] = 'AIRMASS.WMEAN_EQU'
        self.syst['Seeing'] = 'FWHM.WMEAN_EQU'
        self.syst['Zero point residuals'] = 'SIGMA_MAG_ZERO.QSUM_EQU'
        self.syst['Sky variance'] = 'SKYVAR_WMEAN_EQU'
        self.syst['Effective exposure time (mean)'] = 'T_EFF.WMEAN_EQU'
        self.syst['Effective exposure time (sum)'] = 'T_EFF_EXPTIME.SUM_EQU'

        # self.templates_dir = {}
        # self.templates = []

        for band in self.bands:
            for key, name in self.syst.items():
                tempname = 'y3a2_{}_o.4096_t.32768_{}._nside{}.fits'.format(band, name, self.nside)
                temp = hp.read_map(os.path.join(templates_dir, tempname), verbose=False)
                self.templates_dir['%s [%s band]'%(key, band)] = temp
                # self.templates.append(temp)

        # self.templates = np.array(self.templates)
        # self.templates = np.expand_dims(self.templates, axis=1)

    def load_PSF(self, hm, PSF_dir, make_fields=True):
        raise NotImplementedError
        # psf_maps = {}

        # keys = ['obs_e1', 'obs_e2', 'piff_e1', 'piff_e2', 'mask']
        # for k in keys:
        #     psf_maps[k] = hp.read_map(os.path.join(PSF_dir, 'PSF_%s_nside%i.fits'%(k, self.nside)), verbose=False)

        # self.psf_maps = {}
        # self.psf_maps['obs'] = [psf_maps['obs_e1'], -1.0*psf_maps['obs_e2']]
        # self.psf_maps['piff'] = [psf_maps['piff_e1'], -1.0*psf_maps['piff_e2']]
        # self.psf_maps['res'] = [psf_maps['obs_e1']-psf_maps['piff_e1'], -1.0*(psf_maps['obs_e2']-psf_maps['piff_e2'])]

        # self.psf_mask_apo = nmt.mask_apodization(psf_maps['mask'], aposize=hm.aposize, apotype=hm.apotype)

        # self.psf_fields = {}
        # for k in self.psf_maps.keys():
        #     if make_fields:
        #         self.psf_fields[k]  = nmt.NmtField(self.psf_mask_apo, self.psf_maps[k], purify_e=hm.purify_e, purify_b=hm.purify_b)
        #     else:
        #         self.psf_fields[k]  = None
        # # self.psf_fields['obs']  = nmt.NmtField(psf_mask_apo, [self.psf_maps['obs_e1'], -1.0*self.psf_maps['obs_e2']], purify_e=hm.purify_e, purify_b=hm.purify_b)
        # # self.psf_fields['piff'] = nmt.NmtField(psf_mask_apo, [self.psf_maps['piff_e1'], -1.0*self.psf_maps['piff_e2']], purify_e=hm.purify_e, purify_b=hm.purify_b)
        # # self.psf_fields['res']  = nmt.NmtField(psf_mask_apo, [self.psf_maps['obs_e1']-self.psf_maps['piff_e1'], -1.0*(self.psf_maps['obs_e1']-self.psf_maps['piff_e2'])], purify_e=hm.purify_e, purify_b=hm.purify_b)
        # #
        # # return psf_maps

    def _compute_random_auto_cls(self, hm, ibin, nrandom=0, save_wsp=True):
        raise NotImplementedError

    def _compute_cross_template_cls(self, hm, ibin, nrandom=0, save=True):
        raise NotImplementedError

    def _compute_cross_PSF_cls(self, hm, ibin, nrandom=0, save=True):
        raise NotImplementedError

    def plot_cls(self, hm, cls, nrows, ncols, figname='', titles=None, ylabels=None,
                    showy0=False, symy0=False, chi2method=None, blindyaxis=False, fig=None, return_fig=False,
                    factor_ell=0, c=None, ls=None, pdf=False, xlim=None, ylim=None, xscale='linear', yscale='linear'):
        if fig is None:
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, 3*nrows))
        else:
            axes = np.array(fig.axes)
        axes = np.array(axes).reshape((nrows,ncols))

        ell = hm.b.get_effective_ells()

        if factor_ell == 1:
            factor = ell
        elif factor_ell == 2:
            factor = ell*(ell+1.)
        else:
            factor = 1.
        
        chi2 = {}
        for i in range(nrows):
            for j in range(ncols):
                ax = axes[i,j]
                if showy0:
                    ax.axhline(y=0, c='0.8', lw=1)
                y = cls[(i,j)]
                if 'random' in y.keys():
                    nrandom = y['random'].shape[0]
                    for r in range(nrandom):
                        ax.plot(ell, factor*y['random'][r], c='r', alpha=max(0.01, 1./nrandom))
                if chi2method is not None:
                    _chi2, _pval = get_chi2(y['data'], y['random'], smooth=(chi2method=='smooth'), return_pval=True)
                    label = '$\\chi^2_{{{:}}} = {:.2f}$ ($p={:.2g}$)'.format(len(ell), _chi2, _pval)
                    chi2[(i,j)] = _chi2
                else:
                    label = None
                ax.plot(ell, factor*y['data'], label=label, c='b' if c is None else c, ls='-' if ls is None else ls)
                if titles is not None:
                    ax.set_title(titles[(i,j)], fontsize=8)
                ax.set_xlabel('$\\ell$')
                if xlim is not None:
                    ax.set_xlim(xlim)
                # else:
                #     ax.set_xlim(0, hm.b.lmax)
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)

                if ylim is not None:
                    ax.set_ylim(ylim)

                if ylabels is not None:
                    ax.set_ylabel(ylabels[(i,j)])

                if symy0:
                    vmax = max(np.abs(ax.get_ylim()))
                    ax.set_ylim(-vmax,+vmax)

                if chi2method is not None:
                    ax.legend(loc=1)

                if blindyaxis:
                    ax.set_yticks([])
        plt.tight_layout()

        make_directory(self.config.path_figures+'/'+self.name)
        figfile = os.path.join(self.config.path_figures, self.name, 'cls_{}_{}_{}_{}_nside{}.{}'.format(figname, self.obs_name, self.config.name, self.mode, self.nside, 'pdf' if pdf else 'png'))
        plt.savefig(figfile, dpi=300)

        out = []
        if return_fig:
            out.append(fig)
        if chi2method is not None:
            out.append(chi2)
        return tuple(out)


