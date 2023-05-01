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
        
    __version__ = '0.0.3'
    
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

            self.fill_result_i(result_buffer[event_i], peaks_in_event_i, _['s2_bi_match'], _['s2_po_match'])


    def fill_result_i(self, event, peaks, bi_i, po_i):
            """For a single event with the result_buffer"""

            if (bi_i>=0) and (po_i>=0):

                largest_s2s, s2_idx = self.get_largest_sx_peaks(peaks, s_i=2, number_of_peaks=0)

                bipo_mask = [bi_i, po_i]
                s2_idx = s2_idx[bipo_mask]
                largest_s2s = largest_s2s[bipo_mask]

                largest_s1s, s1_idx = self.get_largest_sx_peaks(
                    peaks,
                    s_i=1,
                    number_of_peaks=2)

                largest_s1s = largest_s1s[::-1]
                s1_idx = s1_idx[::-1]
                
                self.set_sx_index(event, s1_idx, s2_idx)
                self.set_event_properties(event, largest_s1s, largest_s2s, peaks)

                # Loop over S1s and S2s and over main / alt.
                for s_i, largest_s_i in enumerate([largest_s1s, largest_s2s], 1):
                    # Largest index 0 -> main sx, 1 -> alt sx
                    for largest_index, main_or_alt in enumerate(['s', 'alt_s']):
                        peak_properties_to_save = [name for name, _, _ in self.peak_properties]
                        if s_i == 2:
                            peak_properties_to_save += ['x', 'y']
                            peak_properties_to_save += self.posrec_save
                        field_names = [f'{main_or_alt}{s_i}_{name}' for name in peak_properties_to_save]
                        self.copy_largest_peaks_into_event(event,
                                                        largest_s_i,
                                                        largest_index,
                                                        field_names,
                                                        peak_properties_to_save)


