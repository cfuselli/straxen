import strax
import numpy as np
import numba
import straxen


export, __all__ = strax.exporter()

@export
class BiPoVariables(strax.Plugin):
    """
    Compute:
    - peak properties
    - peak positions
    of the first three main (in area) S1 and ten S2.
    
    The standard PosRec algorithm and the three different PosRec algorithms (mlp, gcn, cnn)
    are given for the five S2.
    """
        
    __version__ = '4.0.0'
    
    depends_on = ('events',
                  'peak_basics',
                  'peak_positions',
                  'peak_proximity')
    
    # TODO change name
    provides = 'bi_po_variables'
    data_kind = 'events'
    loop_over = 'events'
    
    max_n_s1 = straxen.URLConfig(default=3, infer_type=False,
                                    help='Number of S1s to consider')

    max_n_s2 = straxen.URLConfig(default=10, infer_type=False,
                                    help='Number of S2s to consider')


    peak_properties = (
        # name                dtype       comment
        ('time',              np.int64,   'start time since unix epoch [ns]'),
        ('center_time',       np.int64,   'weighted center time since unix epoch [ns]'),
        ('endtime',           np.int64,   'end time since unix epoch [ns]'),
        ('area',              np.float32, 'area, uncorrected [PE]'),
        ('n_channels',        np.int32,   'count of contributing PMTs'),
        ('n_competing',       np.float32, 'number of competing PMTs'),
        ('max_pmt',           np.int16,   'PMT number which contributes the most PE'),
        ('max_pmt_area',      np.float32, 'area in the largest-contributing PMT (PE)'),
        ('range_50p_area',    np.float32, 'width, 50% area [ns]'),
        ('range_90p_area',    np.float32, 'width, 90% area [ns]'),
        ('rise_time',         np.float32, 'time between 10% and 50% area quantiles [ns]'),
        ('area_fraction_top', np.float32, 'fraction of area seen by the top PMT array')
        )

    pos_rec_labels = ['cnn', 'gcn', 'mlp']

    def setup(self):

        self.posrec_save = [(xy + algo, xy + algo) for xy in ['x_', 'y_'] for algo in self.pos_rec_labels] # ???? 
        self.to_store = [name for name, _, _ in self.peak_properties]

    def infer_dtype(self):
                
        # Basic event properties  
        basics_dtype = []
        basics_dtype += strax.time_fields
        basics_dtype += [('n_peaks', np.int32, 'Number of peaks in the event'),
                        ('n_incl_peaks_s1', np.int32, 'Number of included S1 peaks in the event'),
                        ('n_incl_peaks_s2', np.int32, 'Number of included S2 peaks in the event')]

        # For S1s and S2s
        for p_type in [1, 2]:
            if p_type == 1:
                max_n = self.max_n_s1
            if p_type == 2:
                max_n = self.max_n_s2
            for n in range(max_n):
                # Peak properties
                for name, dt, comment in self.peak_properties:
                    basics_dtype += [(f's{p_type}_{name}_{n}', dt, f'S{p_type}_{n} {comment}'), ]                

                if p_type == 2:
                    # S2 Peak positions
                    for algo in self.pos_rec_labels:
                        basics_dtype += [(f's2_x_{algo}_{n}', 
                                          np.float32, f'S2_{n} {algo}-reconstructed X position, uncorrected [cm]'),
                                         (f's2_y_{algo}_{n}',
                                          np.float32, f'S2_{n} {algo}-reconstructed Y position, uncorrected [cm]')]

        return basics_dtype

    @staticmethod
    def set_nan_defaults(buffer):
        """
        When constructing the dtype, take extra care to set values to
        np.Nan / -1 (for ints) as 0 might have a meaning
        """
        for field in buffer.dtype.names:
            if np.issubdtype(buffer.dtype[field], np.integer):
                buffer[field][:] = -1
            else:
                buffer[field][:] = np.nan

    @staticmethod
    def get_largest_sx_peaks(peaks,
                             s_i,
                             number_of_peaks=2):
        """Get the largest S1/S2. For S1s allow a min coincidence and max time"""
        # Find all peaks of this type (S1 or S2)

        s_mask = peaks['type'] == s_i
        selected_peaks = peaks[s_mask]
        s_index = np.arange(len(peaks))[s_mask]
        largest_peaks = np.argsort(selected_peaks['area'])[-number_of_peaks:][::-1]
        return selected_peaks[largest_peaks], s_index[largest_peaks]
            

    def compute(self, events, peaks):

        result = np.zeros(len(events), dtype=self.dtype)
        self.set_nan_defaults(result)

        split_peaks = strax.split_by_containment(peaks, events)

        result['time'] = events['time']
        result['endtime'] = events['endtime']

        for event_i, _ in enumerate(events):

            peaks_in_event_i = split_peaks[event_i]

            largest_s1s, s1_idx = self.get_largest_sx_peaks(peaks_in_event_i, s_i=1, number_of_peaks=self.max_n_s1)
            largest_s2s, s2_idx = self.get_largest_sx_peaks(peaks_in_event_i, s_i=2, number_of_peaks=self.max_n_s2)

            for i, p in enumerate(largest_s1s): 
                for prop in self.to_store:
                    result[event_i][f's1_{prop}_{i}'] = p[prop]

            for i, p in enumerate(largest_s2s):
                for prop in self.to_store:
                    result[event_i][f's2_{prop}_{i}'] = p[prop]  
                for name_alg in self.posrec_save:
                    result[event_i][f's2_{name_alg[0]}_{i}'] = p[name_alg[1]]

        return result