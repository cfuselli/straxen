import sys
sys.path.insert(0, '/daq_common/carlo/test_software/straxen_software_veto')


import strax
import straxen
from straxen.plugins.raw_records_sv import software_veto
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

import strax
import straxen

"""


_plugins = []

for name, pl in registry:
    if pl == straxen.DAQReader:
        pass
    elif pl == straxen.Fake1TDAQReader:
        pass
    elif pl.__base__ == straxen.plugins.raw_records_sv._software_veto_base.RawRecordsSoftwareVetoBase:
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
            data_kind = {t.replace('_sv','')+'_sv':init_pl.data_kind[t] for t in init_pl.data_kind}
        else:
            data_kind = "'"+str(init_pl.data_kind+'_sv')+"'"
            
        if isinstance(pl.save_when, immutabledict):
            save_when = str(immutabledict({t.replace('_sv','')+'_sv':init_pl.save_when[t] for t in pl.save_when}))
            for k, v in save_when_replacers.items():
                save_when = save_when.replace(k, v)           
    
        if isinstance(pl.rechunk_on_save, immutabledict):
            rechunk_on_save = str(immutabledict({t.replace('_sv','')+'_sv':init_pl.rechunk_on_save[t] for t in pl.rechunk_on_save}))
            for k, v in rechunk_replacers.items():
                rechunk_on_save = rechunk_on_save.replace(k, v)
                
        if init_pl.multi_output:
            dtype = str({prov+'_sv': init_pl.dtype_for(prov) for prov in pl.provides})
            dtype = dtype.replace('dtype(', '').replace(')])', ')]')
        else:
            dtype = init_pl.dtype_for(name)
            
        def new_infer_dtype(self):
            return self.dtype


        
        classtofile = f"""

class {pl.__name__}SV(straxen.{pl.__name__}):
    depends_on = {depends_on}
    provides = {provides}
    dtype = {dtype}
    data_kind = {data_kind}

    def infer_dtype(self):
        return self.dtype

"""
        tofile += classtofile
        
        if isinstance(pl.save_when, immutabledict):
            tofile += f"""
    save_when = {save_when}
"""
    
        if isinstance(pl.rechunk_on_save, immutabledict):
            tofile += f"""
    rechunk_on_save = {rechunk_on_save}
"""

    _plugins.append(pl)


with open(os.path.join(_dir, '_software_veto_copies.py'), "w") as text_file:
    text_file.write(tofile)

    
print('Finished')