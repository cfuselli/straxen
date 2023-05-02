import strax
import numpy as np
import numba
import straxen
from scipy.optimize import curve_fit

export, __all__ = strax.exporter()

@export
class EventBasicsMultiS2ShapeFit(strax.Plugin):
    """
    Compute:

    """
        
    __version__ = '4.0.0'
    
    depends_on = ('events',
                  'peaks')
    
    # TODO change name
    provides = 'event_basics_multi_s2_shape_fit'
    data_kind = 'events'
    
    max_n_s1 = straxen.URLConfig(default=3, infer_type=False,
                                    help='Number of S1s to consider')

    max_n_s2 = straxen.URLConfig(default=10, infer_type=False,
                                    help='Number of S2s to consider')

    peak_properties = (
            # name                dtype       comment
            ('chi2', np.float32, 'chi2'),
            ('mean', np.float32, 'mean'),
            ('sigma', np.float32, 'sigma'),
            ('area', np.float32, 'area of fitted gaussian')
            )
    

    def infer_dtype(self):
                
        # Basic event properties  
        basics_dtype = []
        basics_dtype += strax.time_fields

        # For S2s
        p_type = 2
        max_n = self.max_n_s2
        for n in range(max_n):
            # Peak properties
            for name, dt, comment in self.peak_properties:
                basics_dtype += [(f's{p_type}_fit_{name}_{n}', dt, f'S{p_type}_{n} fit {comment}'), ]                

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
    
    @staticmethod
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    def compute(self, events, peaks):

        result = np.zeros(len(events), dtype=self.dtype)
        self.set_nan_defaults(result)
        split_peaks = strax.split_by_containment(peaks, events)

        result['time'] = events['time']
        result['endtime'] = events['endtime']

        for event_i, _ in enumerate(events):

            peaks_in_event_i = split_peaks[event_i]

            largest_s2s, s2_idx = self.get_largest_sx_peaks(peaks_in_event_i, s_i=2, number_of_peaks=self.max_n_s2)

            for i, p in enumerate(largest_s2s):

                # Define the data to be fit
                x = np.arange(200)[:p['length']]
                y = p['data'][:p['length']]/max(p['data'])  # assuming you want to fit the first peak

                # Fit the data with one Gaussian                
                try:
                    popt, pcov = curve_fit(self.gaussian, x, y, p0=[.1, 80, 10])
                except Exception as e:
                    popt = [0,0,0]
                
                # Calculate the chi-square goodness of fit statistic
                residuals = y - self.gaussian(x, *popt)
                chi2 = np.sum(residuals**2)
                chi2_red = chi2 / (len(x) - len(popt))


                result[event_i][f's2_fit_chi2_{i}']  = chi2_red
                result[event_i][f's2_fit_mean_{i}']  = popt[1]
                result[event_i][f's2_fit_sigma_{i}'] = popt[2]
                result[event_i][f's2_fit_area_{i}']  = max(p['data'])*popt[0]*sigma/(1/np.sqrt(2*np.pi))

        return result



