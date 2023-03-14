import strax
import straxen
from immutabledict import immutabledict
from straxen.plugins.raw_records_sv._software_veto_base import RawRecordsSoftwareVetoBase

export, __all__ = strax.exporter()

@export
class DummyVeto(RawRecordsSoftwareVetoBase):
    """
    Test software veto framework: keep eveything
    """

    __version__ = 'dummy-veto-0.0.1'

    def software_veto_mask(self, e):
        
        m = e['time'] > 0 
        
        return m

@export
class RadialVeto(RawRecordsSoftwareVetoBase):
    """
    Radial sofrtare veto 
    Deletes raw records of events outside certain r
    """

    __version__ = 'radial-veto-0.0.1'

    def software_veto_mask(self, e):
        
        m = (e['x']**2 + e['y']**2) > 50**2
        
        return m

@export
class HighEnergyVeto(RawRecordsSoftwareVetoBase):
    """
    High energy sofrtare veto 
    Deletes raw records for events with high s1 and s2 area
    """

    __version__ = 'high-energy-veto-0.0.1'

    def software_veto_mask(self, e):
        
        m = (e['s1_area'] > 1000) & (e['s2_area'] > 100000)
        
        return m

@export
class ExamplePeakLevel(RawRecordsSoftwareVetoBase):
    """
    Example veto on peak level, needs to specify veto_mask_on
    """

    __version__ = 'example-peak-level-0.0.2'
    depends_on = ('raw_records', 'raw_records_aqmon', 'peak_basics')    

    def compute(self, raw_records, raw_records_aqmon, peaks):
        # base class written to work on events, but we just care about the time intervals
        return super().compute(raw_records, raw_records_aqmon, events=peaks)
	
    def software_veto_mask(self, p):
        
        m = (p['type'] == 2) & (p['area'] > 100000)
        
        return m