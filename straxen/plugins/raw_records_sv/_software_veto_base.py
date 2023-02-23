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
    
    # if extra dependency on i.e. peaks_proximity is needed, 
    # redefine the depends_on in the software_vet0.py plugin (see ExamplePeakLevel)
    # keeping the order raw_records, raw_records_aqmon, peaks, events
    depends_on = ('raw_records', 'raw_records_aqmon', 'peak_basics', 'event_info')

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

    # can be either 'events' or 'peaks'
    veto_mask_on = 'events'

    software_veto_touching_window = straxen.URLConfig(
        default=int(0), infer_type=False,
        help='Strax touching window for container and thing (raw_records and events).')

    software_veto_pre_scaling = straxen.URLConfig(
        default=int(0), infer_type=False,
        help='This sets the pre_scaling factor (keep a fracion of the events we want to delete)'
             '  0   to delete all non-wanted raw_records'
             '  0.5 to keep half of the non-wanted raw_records'
             '  1   the software veto is basically deactivated')

    def infer_dtype(self):
        return {
            d: strax.raw_record_dtype(
                samples_per_record=self.config["record_length"])
            for d in self.provides}
    
    def software_veto_mask(self, objects):
                
        return NotImplementedError("""
            This is a base plugin, 
            please build a plugin with this function""")
            
    def compute(self, raw_records, raw_records_aqmon, peaks, events):
            
        result = dict()

        # define events of which to delete raw_records
        if self.veto_mask_on == 'events':
            objects_to_delete = events[self.software_veto_mask(events)]
        elif self.veto_mask_on == 'peaks':
            objects_to_delete = peaks[self.software_veto_mask(peaks)]
        else:
            return NotImplementedError("Currently possible to cut on events or peaks data-kinds only")

        # apply pre-scaling and update objects to delete
        r = np.random.random(len(objects_to_delete))
        pre_scaling_mask = (r>self.software_veto_pre_scaling)
        objects_to_delete = objects_to_delete[pre_scaling_mask]

        # get mask of raw_records to delete
        veto_mask = self.get_touching_mask(raw_records, objects_to_delete)
        
        # Result: raw_records to keep
        result[self.provides[0]] = raw_records[veto_mask]

        # Result: aqmon to add
        result[self.provides[1]] = strax.sort_by_time(
            np.concatenate([
                raw_records_aqmon,
                self._software_veto_time(
                    start=objects_to_delete['time'],
                    end=strax.endtime(objects_to_delete),
                    dt=raw_records[0]['dt']
                    )]))

        return result
    
    def get_touching_mask(self, things, containers):
        
        # things = raw_records
        # containers = i.e. events

        # start with keep everything
        mask = np.full(len(things), True)

        # throw away things inside every container 
        for i0, i1 in strax.touching_windows(things, containers, window=self.software_veto_touching_window):
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
