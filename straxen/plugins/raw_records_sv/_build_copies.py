import strax
import straxen
from immutabledict import immutabledict
import sys, os

_dir = os.path.dirname(os.path.abspath(__file__))
print(_dir)

st = straxen.contexts.xenonnt_online(output_folder='')
registry = st._plugin_class_registry.copy().items()

save_when_replacers = {"'":'',
             '<':'',
             '>':'',
             '{':'',
             '}':'',
             '0':'',
             '1':'',
             '2':'',
             '3':'',
             '4':'',
             ': ':'',
             'Save':'=strax.Save',
            }

rechunk_replacers = {"'":'',
             '<':'',
             '>':'',
             '{':'',
             '}':'',
             ':':'='
            }


tofile = """
from immutabledict import immutabledict
import numba
import numpy as np
import inspect
import strax
import straxen

"""


_plugins = []

for name, pl in registry:
    if pl == straxen.DAQReader:
        pass
    elif pl == straxen.Fake1TDAQReader:
        pass
    elif name.endswith('_sv'): # ext_timings_nv_sv_sv ?? 
        pass
    elif pl in _plugins:
        pass
    else:
                
        print(name)

        # I need initialisation to save myself
        init_pl = st.get_single_plugin('000000', name)

        provides = [prov.replace('_sv','')+'_sv' for prov in strax.to_str_tuple(pl.provides)]
        depends_on = [deps.replace('_sv','')+'_sv' for deps in strax.to_str_tuple(pl.depends_on)]

        if isinstance(init_pl.data_kind, dict):
            data_kind = {t.replace('_sv','')+'_sv':init_pl.data_kind[t]+'_sv' for t in init_pl.data_kind}
        else:
            data_kind = "'"+str(init_pl.data_kind+'_sv')+"'"
            
        if isinstance(pl.save_when, immutabledict):
            save_when = str(immutabledict({t.replace('_sv','')+'_sv':init_pl.save_when[t] for t in pl.save_when}))
            for k, v in save_when_replacers.items():
                save_when = save_when.replace(k, v)
            save_when = f"""
    save_when = {save_when}
    """
        else:
            save_when = ''
    
        if isinstance(pl.rechunk_on_save, immutabledict):
            rechunk_on_save = str(immutabledict({t.replace('_sv','')+'_sv':init_pl.rechunk_on_save[t] for t in pl.rechunk_on_save}))
            for k, v in rechunk_replacers.items():
                rechunk_on_save = rechunk_on_save.replace(k, v)
            rechunk_on_save = f"""
    rechunk_on_save = {rechunk_on_save}
    """
        else:
            rechunk_on_save = ''
        
        if init_pl.multi_output:
            dtype = str({prov+'_sv': init_pl.dtype_for(prov) for prov in pl.provides})
            dtype = dtype.replace('dtype(', '').replace(')])', ')]')
        else:
            dtype = init_pl.dtype_for(name)

        compute_takes_chunk_i   = init_pl.compute_takes_chunk_i
        compute_takes_start_end =  init_pl.compute_takes_start_end

        if init_pl.multi_output:
            output = """

        p_mapping = {{v: k for k, v in zip(strax.to_str_tuple(self.provides), 
                                        strax.to_str_tuple(super().provides))}}
        return {{p_mapping[k]: v for k,v in result.items()}}
"""
        else:
            output = """
        return result
"""
        



        
        classtofile = f"""

class {pl.__name__}SV(straxen.{pl.__name__}):
    depends_on = {depends_on}
    provides = {provides}
    dtype = {dtype}
    data_kind = {data_kind}
    {save_when}
    {rechunk_on_save}

    def __init__(self):
        super().__init__()
        self.compute_takes_chunk_i = {compute_takes_chunk_i}
        self.compute_takes_start_end = {compute_takes_start_end}

    def infer_dtype(self):
        super().infer_dtype()
        return self.dtype

    def compute(self, **kwargs):
        
        _kwargs = {{}}
        for k,v in kwargs.items():
            if k not in ['chunk_i', 'end', 'start']:
                _kwargs[k.replace('_sv', '')] = v
            else:
                _kwargs[k] = v

        result = super().compute(**_kwargs)

        {output}

"""

    tofile += classtofile
    _plugins.append(pl)


with open(os.path.join(_dir, '_software_veto_copies.py'), "w") as text_file:
    text_file.write(tofile)

    
print('Finished')