from .observable import *

class Template(Observable):
    def __init__(self, config, nside, mode, nzbins, obs_name, map_name, *args, **kwargs):
        super(Template, self).__init__(config, nside, mode, nzbins, obs_name, list(map_name), *args, **kwargs)
        self.map_name = self.map_names[0]
        # self.beam = beam

    def add_maps(self, ibin, map=None, mask=None):
        if isinstance(map, str): #then it's a file name
            assert isinstance(mask, str)
            self.maps[ibin][self.map_name] = hp.ud_grade(hp.read_map(map), nside_out=self.nside)
            self.masks[ibin] = hp.ud_grade(hp.read_map(mask), nside_out=self.nside)
        else:
            self.maps[ibin][self.map_name] = hp.ud_grade(map, nside_out=self.nside)
            self.masks[ibin] = hp.ud_grade(mask, nside_out=self.nside)

    def prepare_fields(self):
        out = {}
        for ibin in self.zbins:
            out[ibin] = [self.maps[ibin][self.map_name]]
        return out

    # def make_fields(self, hm, include_templates=True):
    #     for ibin in tqdm(self.zbins, desc='{}.make_fields'.format(self.obs_name)):
    #         self.fields[ibin] = nmt.NmtField(self.masks_apo[ibin], [self.maps[ibin][self.map_name]], beam=self.beam, **hm.fields_kw)

