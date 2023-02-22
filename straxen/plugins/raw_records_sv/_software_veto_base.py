import numpy as np
import strax
import straxen
from immutabledict import immutabledict

export, __all__ = strax.exporter()

from straxen.plugins.raw_records.daqreader import SOFTWARE_VETO_CHANNEL

@export
@strax.takes_config(
    # DAQ settings -- should match settings given to redax
    strax.Option('record_length', default=110, track=False, type=int,
                 help="Number of samples per raw_record"),
    )
class RawRecordsSoftwareVetoBase(strax.Plugin):
    
    """
    Software veto for raw records - yes, we throw them away forever!

    contact: Carlo Fuselli (cfuselli@nikhef.nl)
    """
        
    __version__ = '0.0.5'
    
    depends_on = ('raw_records', 'raw_records_aqmon', 'event_info')

    provides = (
        'raw_records_sv',
        'raw_records_aqmon_sv',
    )

    data_kind = immutabledict(zip(provides, provides))

    rechunk_on_save = immutabledict(
        raw_records_sv=False,
        raw_records_aqmon_sv=True,
    )

    parallel = 'process'
    chunk_target_size_mb = 50
    compressor = 'lz4'
    input_timeout = 300

    # TODO test with window > 0 
    window = 0 # ns (should pass as option)
    
    def infer_dtype(self):
        return {
            d: strax.raw_record_dtype(
                samples_per_record=self.config["record_length"])
            for d in self.provides}
    
    def software_veto_mask(self, e):
                
        return NotImplementedError("""
            This is a base plugin, 
            please build a plugin with this function""")
            
    def compute(self, raw_records, raw_records_aqmon, events):
            
        result = dict()
        dt = raw_records[0]['dt']
        events_to_delete = events[self.software_veto_mask(events)]

        veto_mask = self.get_touching_mask(raw_records, events_to_delete)
        
        # Result: raw_records to keep
        result[self.provides[0]] = raw_records[veto_mask]

        # Result: aqmon to add
        result[self.provides[1]] = strax.sort_by_time(
            np.concatenate([
                raw_records_aqmon,
                self._software_veto_time(
                    start=events_to_delete['time'],
                    end=events_to_delete['endtime'],
                    dt=dt
                    )]))

        return result
    
    def get_touching_mask(self, things, containers):
        
        # start with keep everything
        mask = np.full(len(things), True)

        # throw away things inside every container 
        for i0, i1 in strax.touching_windows(things, containers, window=self.window):
            mask[i0:i1] = False

        # return only the things outside the containers
        return mask
    
    def _software_veto_time(self, start, end, dt):
        return strax.dict_to_rec(
            dict(time=start,
                 length=(end - start) // dt,
                 dt=np.repeat(dt, len(start)),
                 channel=np.repeat(SOFTWARE_VETO_CHANNEL, len(start))),
            self.dtype_for(self.provides[0]))
