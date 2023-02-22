import strax
import straxen
from immutabledict import immutabledict
from straxen.plugins.raw_records_sv._software_veto_base import RawRecordsSoftwareVetoBase

export, __all__ = strax.exporter()

@export
class RadialVeto(RawRecordsSoftwareVetoBase):
    """
    Radial sofrtare veto 
    Deletes raw records of events outside certain r
    """

    __version__ = '0.0.1'

    def software_veto_mask(self, e):
        
        m = (e['x']**2 + e['y']**2) > 50**2
        
        return m

@export
class HighEnergyVeto(RawRecordsSoftwareVetoBase):
    """
    High energy sofrtare veto 
    Deletes raw records for events with high s1 and s2 area
    """

    __version__ = '0.0.1'

    def software_veto_mask(self, e):
        
        m = (e['s1_area'] > 1000) & (e['s2_area'] > 100000)
        
        return m