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
from .observable import Observable
import pickle
import numpy as np
import itertools
import warnings
import twopoint

class Harmony(object):
    def __init__(self, config, nside, aposize=2.0, apotype='C1', b=None, nlb=None, nproc=0, **kwargs):
        self.config = config
        self.name = config.name
        self.nside = nside

        self.aposize = aposize
        self.apotype = apotype

        self.purify_e = kwargs.get('purify_e', False)
        self.purify_b = kwargs.get('purify_b', False)
        self.fields_kw = {'purify_e':self.purify_e, 'purify_b':self.purify_b}

        if b is None:
            self.nlb = nlb
            self.b = nmt.NmtBin(self.nside, nlb=nlb)
        else:
            self.b = b
            
        # self.lmax = b.lmax
        self.ell = self.b.get_effective_ells()

        try:
            self.load_cls(kwargs.get('print_summary', True))
        except FileNotFoundError:
            self.cls = {}
            self.cls['ell'] = self.ell
        else:
            assert np.allclose(self.ell, self.cls['ell'])

        self.wsp = {}

        self.nproc = nproc
        if nproc > 1:
            self.pool = multiprocessing.Pool(nproc)
        
        make_directory(self.config.path_output+'/'+self.name)

    def check_update_cls(self, obs1, obs2):
        key = (obs1.obs_name, obs2.obs_name)
        if key not in self.cls.keys():
            self.cls[key] = {}
            # if i1 is not None:
            #     assert i2 is not None
            #     self.cls[key][(i1,i2)] = {}
        else:
            #print("Replacing cls[%s]".format(str(key)))
            pass
        
        if key not in self.wsp.keys():
            self.wsp[key] = {}
        else:
            #print("Replacing wsp[%s]".format(str(key)))
            pass
    
    def get_workspace_filename(self, obs1, obs2, i1, i2):
        if isinstance(obs1, Observable) and isinstance(obs2, Observable):
            suffix = '{}_{}_{}_{}'.format(obs1.obs_name, obs2.obs_name, i1, i2)
        elif isinstance(obs1, str) and isinstance(obs2, str):
            suffix = '{}_{}_{}_{}'.format(obs1, obs2, i1, i2)
        else:
            raise ValueError
        filename = os.path.join(self.config.path_output, self.name, 'wsp_{}_nside{}_{}.bin'.format(self.config.name, self.nside, suffix))
        return filename

    def load_workspace(self, obs1, obs2, i1, i2, return_wsp=True):
        filename = self.get_workspace_filename(obs1, obs2, i1, i2)
        self.check_update_cls(obs1, obs2)
        if os.path.isfile(filename):
            wsp = nmt.NmtWorkspace()
            wsp.read_from(filename)
            self.wsp[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = wsp
            return wsp
        else:
            raise RuntimeError("Workspace file does not exist.")

    def load_workspace_bpws(self, obs1, obs2, i1, i2):
        filename = self.get_workspace_filename(obs1, obs2, i1, i2)
        filename_bpws = filename[:-4]+'_bpws.npy'

        if os.path.isfile(filename_bpws):
            bpws = np.load(filename_bpws)
        # elif os.path.isfile(filename):
        #     wsp = nmt.NmtWorkspace()
        #     wsp.read_from(filename)
        #     bpws = wsp.get_bandpower_windows()
        else:
            raise RuntimeError("Workspace file does not exist.")
        
        return bpws
    
    def _save_workspace(self, wsp, obs1, obs2, i1, i2, bpws_only=False):
        filename = self.get_workspace_filename(obs1, obs2, i1, i2)
        np.save(filename[:-4]+'_bpws.npy', wsp.get_bandpower_windows())
        if not bpws_only:
            wsp.write_to(filename)

    def prepare_workspace(self, obs1, obs2, i1, i2, save_wsp=True, return_wsp=True):
        assert isinstance(save_wsp, bool) or save_wsp=='bpws_only'

        field1 = obs1.get_field(self, i1)
        field2 = obs2.get_field(self, i2)

        wsp = nmt.NmtWorkspace()
        wsp.compute_coupling_matrix(field1, field2, self.b)

        self.wsp[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = wsp

        if save_wsp:
            self._save_workspace(wsp, obs1, obs2, i1, i2, bpws_only=(save_wsp=='bpws_only'))
        
        if return_wsp:
            return wsp
    
    def prepare_all_workspaces(self, obs1, obs2=None, save_wsp=True):
        if obs2 is None:
            same_obs = True
            obs2 = obs1
        else:
            same_obs = False

        self.check_update_cls(obs1, obs2)
 
        for i1,i2 in tqdm(itertools.product(obs1.zbins, obs2.zbins), desc='Harmony.prepare_all_workspaces [obs1:{}, obs2={}]'.format(obs1.obs_name, obs2.obs_name)):
            if (i2,i1) in self.cls[(obs1.obs_name, obs2.obs_name)].keys() and same_obs:
                # Even if same_obs, order of cls is different ((E1,B2) vs (E2,B1)) so best not to include it.
                # self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = self.cls[(obs1.obs_name, obs2.obs_name)][(i2,i1)]
                continue
            else:
                self.prepare_workspace(obs1, obs2, i1, i2, save_wsp=save_wsp, return_wsp=False)
    
    def get_all_workspaces(self, obs1, obs2=None, save_wsp=True):
        if obs2 is None:
            same_obs = True
            obs2 = obs1
        else:
            same_obs = False

        self.check_update_cls(obs1, obs2)
 
        for i1,i2 in tqdm(itertools.product(obs1.zbins, obs2.zbins), desc='Harmony.get_all_workspaces [obs1:{}, obs2={}]'.format(obs1.obs_name, obs2.obs_name)):
            if (i2,i1) in self.wsp[(obs1.obs_name, obs2.obs_name)].keys() and same_obs:
                # Even if same_obs, order of cls is different ((E1,B2) vs (E2,B1)) so best not to include it.
                # self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = self.cls[(obs1.obs_name, obs2.obs_name)][(i2,i1)]
                continue
            else:
                self.wsp[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = self.get_workspace(obs1, obs2, i1, i2, save_wsp=save_wsp)

    def get_workspace(self, obs1, obs2, i1, i2, save_wsp=True):
        try:
            return self.wsp[(obs1.obs_name, obs2.obs_name)][(i1,i2)]
        except KeyError:
            filename = self.get_workspace_filename(obs1, obs2, i1, i2)
            try:
                wsp = self.load_workspace(obs1, obs2, i1, i2, return_wsp=True)
            except RuntimeError:
                wsp = self.prepare_workspace(obs1, obs2, i1, i2, save_wsp=save_wsp, return_wsp=True)
            return wsp

    def get_workspace_bpws(self, obs1, obs2, i1, i2, save_wsp=True):
        try:
            bpws = self.load_workspace_bpws(obs1, obs2, i1, i2)
            return bpws
        except RuntimeError:
            try:
                wsp = self.get_workspace(obs1, obs2, i1, i2, save_wsp=save_wsp)
                return wsp.get_bandpower_windows()
            except:
                raise RuntimeError("Couldn't load or prepare bpws.")

    def save_cls(self):
        make_directory(self.config.path_output+'/'+self.name)
        filename = os.path.join(self.config.path_output, self.name, 'cls_{}_nside{}.pickle'.format(self.config.name, self.nside))
        pickle.dump(self.cls, open(filename, mode='wb'))

    def load_cls(self, print_summary=True):
        filename = os.path.join(self.config.path_output, self.name, 'cls_{}_nside{}.pickle'.format(self.config.name, self.nside))
        self.cls = pickle.load(open(filename, mode='rb'))

        if print_summary:
            print("Loaded Cl's info:")
            print('  - Multipole bins ({}) = '.format(len(self.cls['ell'])), self.cls['ell'])
            for obs_key in self.cls.keys():
                if obs_key != 'ell':
                    print("  - Observables", obs_key)
                    for bin_key in self.cls[obs_key].keys():
                        print("     - Redshift bins", bin_key)
                        for tr_key, cl in self.cls[obs_key][bin_key].items():
                            print("          -", tr_key, cl.shape)

    def compute_cls(self, obs1, i1, obs2=None, i2=None, save_cls=True, return_cls=True, check_cls=True, save_wsp=True):
        if obs2 is None:
            same_obs = True
            obs2 = obs1
        if i2 is None:
            i2=i1

        if check_cls:
            self.check_update_cls(obs1, obs2)

        field1 = obs1.get_field(self, i1)
        field2 = obs2.get_field(self, i2)

        wsp = self.get_workspace(obs1, obs2, i1, i2, save_wsp=save_wsp)

        _cls = compute_master(field1, field2, wsp)

        if (i1, i2) not in self.cls[(obs1.obs_name, obs2.obs_name)].keys():
            self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = {}

        self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)]['true'] = _cls

        if save_cls:
            self.save_cls()
        
        if return_cls:
            return self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)]

    def compute_all_cls(self, obs1, obs2=None, save_cls=True, save_wsp=True):
        if obs2 is None:
            same_obs = True
            obs2 = obs1
        else:
            same_obs = False

        self.check_update_cls(obs1, obs2)

        for i1,i2 in tqdm(itertools.product(obs1.zbins, obs2.zbins), desc='Harmony.compute_all_cls [obs1:{}, obs2={}]'.format(obs1.obs_name, obs2.obs_name)):
            if (i2,i1) in self.cls[(obs1.obs_name, obs2.obs_name)].keys() and same_obs:
                # Even if same_obs, order of cls is different ((E1,B2) vs (E2,B1)) so best not to include it.
                # self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = self.cls[(obs1.obs_name, obs2.obs_name)][(i2,i1)]
                continue
            else:
                self.compute_cls(obs1=obs1, i1=i1, obs2=obs2, i2=i2, save_cls=save_cls, return_cls=False, save_wsp=save_wsp)

    def compute_random_auto_cls(self, obs, ibins=None, nrandom=1, save_cls=True, save_wsp=True):
        if ibins is None:
            ibins = obs.zbins
        else:
            ibins = np.array(ibins)
        for ibin in tqdm(ibins, desc='Harmony.compute_random_auto_cls [obs:{}]'.format(obs.obs_name)):
            clr = obs._compute_random_auto_cls(self, ibin, nrandom, save_wsp=save_wsp)
            if 'random' in self.cls[(obs.obs_name, obs.obs_name)][(ibin,ibin)].keys():
                # print("Adding to existing randoms.", self.cls[(obs.obs_name, obs.obs_name)][(ibin,ibin)]['random'].shape, clr.shape)
                self.cls[(obs.obs_name, obs.obs_name)][(ibin,ibin)]['random'] = np.concatenate([self.cls[(obs.obs_name, obs.obs_name)][(ibin,ibin)]['random'], clr])
                # print(self.cls[(obs.obs_name, obs.obs_name)][(ibin,ibin)]['random'].shape)
            else:
                self.cls[(obs.obs_name, obs.obs_name)][(ibin,ibin)]['random'] = clr
            if save_cls:
                self.save_cls()



    # def compute_all_auto_cls(self, obs, nrandom=0, save_cls=True, save_wsp=True):
    #     self.check_update_cls(obs, obs)

    #     for ibin in tqdm(obs.zbins, desc='Harmony.compute_all_auto_cls [obs:{}]'.format(obs.obs_name)):
    #         self.cls[(obs.obs_name, obs.obs_name)][(ibin,ibin)] = obs._compute_auto_cls(self, ibin, nrandom=nrandom, save_cls=save_cls, save_wsp=save_wsp)

    # def compute_all_cls(self, obs, nrandom=0, save_cls=True, save_wsp=True):
    #     self.check_update_cls(obs, obs)

    #     for i1 in tqdm(obs.zbins, desc='Harmony.compute_all_cls [obs:{}]'.format(obs.obs_name)):
    #         # field1 = obs.get_field(self, i1)
    #         for i2 in obs.zbins:
    #             if (i2,i1) in self.cls[(obs.obs_name, obs.obs_name)].keys():
    #                     # Even if same_obs, order of cls is different ((E1,B2) vs (E2,B1)) so best not to include it.
    #                     # self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)] = self.cls[(obs1.obs_name, obs2.obs_name)][(i2,i1)]
    #                     continue
    #             else:
    #                 if i1 == i2:
    #                     ibin = i1
    #                     self.cls[(obs.obs_name, obs.obs_name)][(ibin,ibin)] = obs._compute_auto_cls(self, ibin, nrandom=nrandom, save_wsp=save_wsp)
    #                     if save_cls:
    #                         self.save_cls()
    #                 else:
    #                     # field2 = obs.get_field(self, i2)
    #                     # self.cls[(obs.obs_name, obs.obs_name)][(i1,i2)] = nmt.compute_full_master(field1, field2, self.b)
    #                     self.compute_cls(obs, obs, i1, i2, save_cls=save_cls, return_cls=False, check_cls=False, save_wsp=save_wsp)

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

    def bin_cl_theory(self, cl_in, obs1, obs2, i1, i2, fix_pixwin=False):
        bpws = self.get_workspace_bpws(obs1, obs2, i1, i2)
        if isinstance(cl_in, dict):
            cl = cl_in[(obs1.obs_name, obs2.obs_name)][(i1,i2)]
        else:
            cl = cl_in

        # return wsp.decouple_cell(wsp.couple_cell(cl_in))
        if fix_pixwin:
            f_ell = hp.pixwin(self.nside)[:bpws.shape[-1]]**2
        else:
            f_ell = np.ones_like(cl[:,:bpws.shape[-1]])

        return np.tensordot(bpws, cl[:,:bpws.shape[-1]]*f_ell, axes=2)
    
    def get_twopoint_filename(self):
        return os.path.join(self.config.path_output, self.name, 'twopoint_cls_{}_nside{}.fits'.format(self.config.name, self.nside))

    def load_twopoint(self):
        filename = self.get_twopoint_filename()
        twopoint.TwoPointFile.from_fits(filename)

    def twopoint_build_spec(self, obs1, obs2, name, kernels, types, spin_idx=0):
        """
        eg name='shear_cl', kernels=('nz_source','nz_source'), types=('GEF','GEF')
        """
        bins1 = []
        bins2 = []
        angbin = []
        value = []
        for (i1,i2), C_ell in self.cls[(obs1.obs_name, obs2.obs_name)].keys():
            cl_temp = C_ell['true'][spin_idx]
            if 'random' in C_ell.keys():
                cl_temp -= np.mean(C_ell['true'][:,spin_idx], axis=0)
            n = len(C_ell)
            bins1 += [i1+1]*n
            bins2 += [i2+1]*n
            angbin += list(np.arange(n))
            value += list(cl_temp)

        spec = twopoint.SpectrumMeasurement(name=name,bins=(np.array(bins1),np.array(bins2)),
                types=types, kernels=kernels, windows='SAMPLE',
                angular_bin=np.array(angbin), value=np.arary(value))
            
        spec.reorder_canonical()

        make_directory(self.config.path_output+'/'+self.name)
        filename = os.path.join(self.config.path_output, self.name, 'twopointspec_{}_nside{}_{}_{}_{}_{}.pickle'.format(self.config.name, self.nside, obs1.obs_name, obs2.obs_name, i1, i2))
        spec.to_fits(filename)

        return spec

    def get_cl(self, obs1, obs2, i1, i2, debias=True):
        out = np.copy(self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)]['true'])
        if obs1==obs2 and i1==i2 and debias:
            out -= np.mean(self.cls[(obs1.obs_name, obs2.obs_name)][(i1,i2)]['random'], axis=0)            
        return out

    def _compute_gaussian_covariance_block(self, obsa1, obsa2, obsb1, obsb2, a1, a2, b1, b2, C_ell, add_noise_to_C_ell=True):
        # warnings.warn("compute_gaussian_covariance is not implemented yet, only returns an array of zeros.")
        # return np.zeros(self.b.get_n_bands(),self.b.get_n_bands())

        def spin_help(obs1, obs2):
            return max(1,obs1.spin)*max(1,obs2.spin)
        n_ell = self.b.get_n_bands()

        def get_C_ell(obs1, obs2, i1, i2, add_noise=add_noise_to_C_ell):
            res = np.copy(C_ell[(obs1.obs_name,obs2.obs_name)][(i1,i2)])
            if add_noise and obs1==obs2 and i1==i2:
                NL = self.b.unbin_cell(self.cls[(obs1.obs_name,obs2.obs_name)][(i1,i2)]['random'].mean(axis=0))
                lowest_ell = self.b.get_ell_list(0)[0]
                for i in range(NL.shape[0]):
                    NL[i,:lowest_ell] = NL[i,lowest_ell]
                res += NL
            return res


        cw=nmt.NmtCovarianceWorkspace()

        suffix = ''
        suffix += obsa1.obs_name+str(a1)+'_'
        suffix += obsa2.obs_name+str(a2)+'_'
        suffix += obsb1.obs_name+str(b1)+'_'
        suffix += obsb2.obs_name+str(b2)+'.pickle'
        filename = os.path.join(self.config.path_output, self.name, 'cw_{}_nside{}_.pickle'.format(self.config.name, self.nside) + suffix)
        if os.path.exists(filename):
            cw.read_from(filename)
        else:
            cw.compute_coupling_coefficients(obsa1.get_field(self, a1),
                                             obsa2.get_field(self, a2),
                                             obsb1.get_field(self, b1),
                                             obsb2.get_field(self, b2)) #<- This is the time-consuming operation

            cw.write_to(filename)

        cov_temp = nmt.gaussian_covariance(cw,
                        obsa1.spin, obsa2.spin, obsb1.spin, obsb2.spin,
                        get_C_ell(obsa1,obsb1,a1,b1),
                        get_C_ell(obsa1,obsb2,a1,b2),
                        get_C_ell(obsa2,obsb1,a2,b1),
                        get_C_ell(obsa2,obsb2,a2,b2),
                        self.get_workspace(obsa1, obsa2, a1, a2),
                        self.get_workspace(obsb1, obsb2, b1, b2))
        
        return cov_temp.reshape([n_ell, spin_help(obsa1, obsa2), n_ell, spin_help(obsb1, obsb2)])#[:, 0, :, 0]
    
    @staticmethod
    def bin_helper(obs, i):
        return obs.zbins.index(i)
    
    def _twopoint_add_bpws(self, obs_pairs, names, spin_idx=0):
        filename = self.get_twopoint_filename()
        hdus = fits.open(filename)
        
        for name, (obs1,obs2) in zip(names, obs_pairs):
            for (i1,i2) in self.cls[(obs1.obs_name, obs2.obs_name)].keys():
                bwps = self.load_workspace_bpws(obs1, obs2, i1, i2)[spin_idx,:,spin_idx,:]
                hdus.append(fits.ImageHDU(bwps, name='bpws_'+name+'_{}_{}'.format(self.bin_helper(obs1,i1)+1,self.bin_helper(obs2,i2)+1)))
        
        hdus.writeto(filename, overwrite=True)

    def build_twopoint(self, obs_pairs, names, kernels, C_ell, spin_idx=0, use_C_ell_as_data=False, add_noise_to_C_ell=True, overwrite=False, clobber=False):
        """
        names is a list of names for each obs pair
        kernels is the list of redshift distribution to be used
        C_ell is a dictionary of theoretical spectra with the same organization as the measured cls [(obs1.obs_name, obs2.obs_name)][(i1,i2)][spin_size,n_ell]
        """
        n_ell = self.b.get_n_bands()

        builder = twopoint.SpectrumCovarianceBuilder()

        # bin_pairs = np.array(self.cls[(obs1.obs_name, obs2.obs_name)].keys())
        # # # sort by bin1, then bin2
        # bin_pairs.view('i8,i8').sort(order=['f0','f1'], axis=0)
        # length = len(bin_pairs) * n

        # Loop over obs pairs, then their redshift bin pais present in cls
        total_size = 0
        ell_b = self.b.get_effective_ells()
        for (obs1, obs2) in tqdm(obs_pairs, desc='Harmony.build_twopoint [data vector]'):
            for (i1,i2) in tqdm(self.cls[(obs1.obs_name, obs2.obs_name)].keys(), desc='[{},{}]'.format(obs1.obs_name, obs2.obs_name)):
                if use_C_ell_as_data:
                    values = self.bin_cl_theory(C_ell, obs1, obs2, i1, i2, fix_pixwin=False)
                else:
                    values = self.get_cl(obs1, obs2, i1, i2, debias=True)
                for i in range(n_ell):
                    builder.add_data_point(kernel1=obs1.kernel,
                                   kernel2=obs2.kernel,
                                   type1=obs1.type,
                                   type2=obs2.type,
                                   bin1=self.bin_helper(obs1,i1)+1,
                                   bin2=self.bin_helper(obs2,i2)+1,
                                   ang=ell_b[i],
                                   angbin=i,
                                   value=values[spin_idx,i])
                    total_size += 1
        covmat = np.zeros((total_size, total_size))

        # Double loop in same order as above
        idxa = 0
        for (obsa1, obsa2) in tqdm(obs_pairs, desc='Harmony.build_twopoint [covariance]'):
            for (a1,a2) in tqdm(self.cls[(obsa1.obs_name, obsa2.obs_name)].keys(), desc='[{},{}]'.format(obsa1.obs_name, obsa2.obs_name)):
                idxb = 0
                for (obsb1, obsb2) in obs_pairs:
                    for (b1,b2) in self.cls[(obsb1.obs_name, obsb2.obs_name)].keys():
                        covmat[idxa:idxa+n_ell,idxb:idxb+n_ell] = self._compute_gaussian_covariance_block(obsa1, obsa2, obsb1, obsb2, a1, a2, b1, b2, C_ell, add_noise_to_C_ell=add_noise_to_C_ell)[:,spin_idx,:,spin_idx]
                        idxb += n_ell
                idxa += n_ell

        builder.set_names(names)
        spectra, covmat_info = builder.generate(covmat, None)
        
        twopointfile = twopoint.TwoPointFile(spectra, kernels, windows='SAMPLE', covmat_info=covmat_info)
        
        filename = self.get_twopoint_filename()
        twopointfile.to_fits(filename, overwrite, clobber)

        self._twopoint_add_bpws(obs_pairs, [names[(obs1.kernel, obs2.kernel, obs1.type, obs2.type)] for obs1,obs2 in obs_pairs], spin_idx=spin_idx)


