import strax
import numpy as np
import numba
import straxen
import itertools

export, __all__ = strax.exporter()

@export
class BiPo214Matching(strax.Plugin):
    """Plugin for matching S2 signals reconstructed as bismuth or polonium peaks in a dataset
    containing BiPo214 events.

    Provides:
    --------
    bi_po_214_matching : numpy.array
        Array containing the indices of the S2 signals reconstructed as bismuth or polonium peaks.
        The index is -9 if the S2 signal is not found, -2 if multiple S2 signals are found,
        or the index of the S2 signal if only one is found.

    Configuration:
    -----------
    tol : int
        Time tolerance window in ns to match S2 signals to BiPo214 events. Default is 3000 ns.
    """
    
    depends_on=('bi_po_variables', )   
    provides = 'bi_po_214_matching'

    __version__ = "2.1.7"
        
    def infer_dtype(self): 
        
        dtype = strax.time_fields + [
                (f's2_bi_match', np.int,
                 f'Index of the S2 reconstructed as a Bismuth peak'),
                (f's2_po_match', np.int,
                 f'Index of the S2 reconstructed as a Polonium peak'),
                (f'n_incl_peaks_s2', np.int,
                 f'S2s considered in the combinations, that passed the mask requirements')]
        
        return dtype

    def setup(self):
        
        self.tol = 3000 # 2 mus tolerance window
        
    def compute(self, events):
        result = np.zeros(len(events), dtype=self.dtype)
        result['time'] = events['time']
        result['endtime'] = events['endtime']
        result['s2_bi_match'] = -9
        result['s2_po_match'] = -9
        
        
        mask = self.box_mask(events)
        
        s2_bi_match, s2_po_match, n_s2s = self.find_match(events[mask], self.tol)
        
        result['s2_bi_match'][mask] = s2_bi_match
        result['s2_po_match'][mask] = s2_po_match
        result['n_incl_peaks_s2'][mask] = n_s2s
        
        return result
        
        
    def find_match(self, events, tol):
        

        # Compute the time difference between the two S1s
        dt_s1 = events['s1_center_time_0'] - events['s1_center_time_1']  
        
        dt_s1_lower = dt_s1 - tol
        dt_s1_upper = dt_s1 + tol
        dt_s1_lower[dt_s1_lower < 0] = 0
        
        # Prepare arrays to store matched S2s
        s2_bi_match = np.full(len(events), -1)
        s2_po_match = np.full(len(events), -1)
        n_s2s       = np.full(len(events), 0)
        
        # Find matching S2s
        for i, event in enumerate(events):
            # Create a list of possible S2 pairs to match
            s2s_idx = self.consider_s2s(event)
            n_s2s[i] = len(s2s_idx)
            possible_pairs = list(itertools.combinations(s2s_idx,2))

            for pair in possible_pairs:
                
                p0, p1 = str(pair[0]), str(pair[1])
                t0, t1 = event['s2_center_time_' + p0], event['s2_center_time_' + p1]
                
                dt_s2 = abs(t0 - t1)
                if dt_s1_lower[i] <= dt_s2 <= dt_s1_upper[i]:
                    if s2_bi_match[i] == -1:
                        s2_bi_match[i] = pair[np.argmin([t0, t1])]
                        s2_po_match[i] = pair[np.argmax([t0, t1])]
                    else:
                        s2_bi_match[i] = -2
                        s2_po_match[i] = -2
                        break
        
        return s2_bi_match, s2_po_match, n_s2s

    @staticmethod
    def consider_s2s(event):
        res = []
        for ip in range(10):
            p = '_'+str(ip)
            consider = True
            consider &= event['s2_area'+p] > 1500                                # low area limit
            consider &= event['s2_area_fraction_top'+p] > 0.5                    # to remove S1 afterpulses
            consider &= np.abs(event['s2_time'+p] - event['s1_time_0']) > 1000  # again to remove afterpulses
            consider &= np.abs(event['s2_time'+p] - event['s1_time_1']) > 1000  # and again to remove afterpulses
            consider &= event['s2_time'+p]-event['s1_time_0'] < 5000000         # 5000mus, S2 is too far in time, not related to Po
            if consider:
                res.append(ip)
        return res

    @staticmethod
    def box_mask(events):
        
        # s1_0 == alpha (Po)
        mask =  (events['s1_area_0'] >  40000) # this is implicit from band cut
        mask &= (events['s1_area_0'] < 120000)    
        mask &= (events['s1_area_fraction_top_0'] < 0.6)

        return mask