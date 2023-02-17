import strax
import straxen
from immutabledict import immutabledict
from straxen.plugins.raw_records_sv._software_veto_base import RawRecordsSoftwareVetoBase

export, __all__ = strax.exporter()

@export
class RawRecordsRadialVeto(RawRecordsSoftwareVetoBase):
    """
    Radial sofrtare veto 
    Deletes raw records of events outside certain r
    """

    provides = (
        'raw_records_sv',
        'raw_records_aqmon_sv',
    )

    data_kind = immutabledict(zip(provides, provides))

    rechunk_on_save = immutabledict(
        raw_records_sv=False,
        raw_records_aqmon_sv=True,
    )

    def software_veto_mask(self, e):
        
        m = (e['x']**2 + e['y']**2) > 50**2
        
        return m