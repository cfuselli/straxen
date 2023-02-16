import numpy as np
import strax
import straxen
from immutabledict import immutabledict

export, __all__ = strax.exporter()

from straxen.plugins.raw_records.daqreader import SOFTWARE_VETO_CHANNEL


@export
@strax.takes_config(

    # All these must have track=False, so the raw_records hash never changes!

    # DAQ settings -- should match settings given to redax
    strax.Option('record_length', default=110, track=False, type=int,
                 help="Number of samples per raw_record"),
    )
class RawRecordsSoftwareVeto(strax.Plugin):
    
    """

    Software veto for raw records
    Depends on event_info

    contact: Carlo Fuselli (cfuselli@nikhef.nl)
    """
        
    __version__ = '0.0.5'
    
    depends_on = ('raw_records', 'raw_records_aqmon', 'event_info')

    provides = (
        'raw_records_sv',
        'raw_records_aqmon_sv',
    )

    data_kind = immutabledict(zip(provides, provides))
    parallel = 'process'
    chunk_target_size_mb = 50
    rechunk_on_save = immutabledict(
        raw_records_sv=False,
        raw_records_aqmon_sv=True,
    )
    compressor = 'lz4'
    input_timeout = 300


    window = 0 # ns? should pass as options
    
    
    def infer_dtype(self):
        return {
            d: strax.raw_record_dtype(
                samples_per_record=self.config["record_length"])
            for d in self.provides}
    
    def software_veto_mask(self, e):
        
        m = (e['x']**2 + e['y']**2) > 50**2
        
        return m
            
    def compute(self, raw_records, raw_records_aqmon, events):
            
        result = dict()

        events_to_delete = events[self.software_veto_mask(events)]

        veto_mask = self.get_touching_mask(raw_records, events_to_delete)
        result['raw_records_sv'] = raw_records[veto_mask]

        dt = raw_records[0]['dt']

        result['raw_records_aqmon_sv'] = self._software_veto_time(
            start=events_to_delete['time'],
            end=events_to_delete['endtime'],
            dt=dt
            )

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
            self.dtype_for('raw_records_sv'))
    
@export
class RawRecordsDownSample(strax.Plugin):
    
    """
    Software data manipulation
    The raw_records of big selected S2s are 'downsampled'
    The lenght of the array stays the same, but with averaged values
    so the data reduction is only at compressed level
    """
        
    __version__ = '0.0.0'
    
    depends_on = ('raw_records', 'peak_basics')
    provides = 'raw_records_down_sample'
    data_kind = 'raw_records_down_sample'

    window = 0 # ns? should pass as options
    downsampling_factor = 5 # must be able to divide 110 with
    
    def infer_dtype(self):

        d = 'raw_records'
        return self.deps[d].dtype_for(d)
    
    def peaks_mask(self, p):
        
        mask  = p['type'] == 2
        mask &= p['area'] > 500000 # PE
        mask &= p['range_50p_area'] > 5000 # ns
        
        return mask
            
    def compute(self, raw_records, peaks):
            
        pp = peaks[self.peaks_mask(peaks)]
        mask = self.get_touching_mask(raw_records, pp)
        
        rr = raw_records.copy()

        rr['data'][mask] = raw_records['data'][mask].reshape(len(raw_records[mask]),
                                                                      -1, 
                                                                      self.downsampling_factor, ).mean(axis=2).repeat(self.downsampling_factor,
                                                                                                                 axis=1)    
        
        return rr
    
    
    def get_touching_mask(self, things, containers):
            
        mask = np.full(len(things), False)

        for i0, i1 in strax.touching_windows(things, containers, window=self.window):
            mask[i0:i1] = True

        return mask
        
