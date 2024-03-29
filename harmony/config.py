
class Config(object):
    def __init__(self, name='name', all_paths=None, path_maps='../maps', path_figures='../figures', path_output='../output', path_wsp='../wsp'):
        self.name = name
        if all_paths is not None:
            self.path_maps = all_paths
            self.path_figures = all_paths
            self.path_output = all_paths
            self.path_wsp = all_paths
        else:
            self.path_maps = path_maps
            self.path_figures = path_figures
            self.path_output = path_output
            self.path_wsp = path_wsp
