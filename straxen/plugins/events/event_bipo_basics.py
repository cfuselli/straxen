import strax
import numpy as np
import numba
import straxen


export, __all__ = strax.exporter()

@export
class EventBiPoBasics(straxen.EventBasics):
    """
    Carlo explain please
    """
        
    __version__ = '0.0.1'
    
    depends_on = ('events',
                  'bi_po_variables',
                  'bi_po_214_matching',
                  'peak_basics',
                  'peak_positions',
                  'peak_proximity')
    
    # TODO change name
    provides = 'event_basics'
    data_kind = 'events'
    loop_over = 'events'
    

    def fill_events(self, result_buffer, events, split_peaks):
        """Loop over the events and peaks within that event"""
        for event_i, _ in enumerate(events):
            peaks_in_event_i = split_peaks[event_i]
            n_peaks = len(peaks_in_event_i)
            result_buffer[event_i]['n_peaks'] = n_peaks

            if not n_peaks:
                raise ValueError(f'No peaks within event?\n{events[event_i]}')

            if event_i['s2_bi_match']>0 and event_i['s2_po_match']>0:
                self.fill_result_i(result_buffer[event_i], peaks_in_event_i)