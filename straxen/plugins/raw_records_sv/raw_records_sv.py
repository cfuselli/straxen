import numpy as np
import strax
import straxen

export, __all__ = strax.exporter()

@export
class RawRecordsSoftwareVeto(strax.Plugin):
    
    """
    Software veto for raw records
    Depends on event_info

    """
        
    __version__ = '0.0.5'
    
    depends_on = ('raw_records', 'event_info')
    provides = 'raw_records_sv'
    data_kind = 'raw_records_sv'

    window = 0 # ns? should pass as options
    
    
    def infer_dtype(self):

        d = 'raw_records'
        return self.deps[d].dtype_for(d)
    
    def software_veto_mask(self, e):
        
        m = (e['x']**2 + e['y']**2) > 50**2
        
        return m
            
    def compute(self, raw_records, events):
            
        ee = events[self.software_veto_mask(events)]
        
        return self.get_touching(raw_records, ee)
    
    
    def get_touching(self, things, containers):
        
        # start with keep everything
        mask = np.full(len(things), True)

        # throw away things inside every container 
        for i0, i1 in strax.touching_windows(things, containers, window=self.window):
            mask[i0:i1] = False

        # return only the things outside the containers
        return things[mask]
    
    
    
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
        
