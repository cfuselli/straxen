import strax
import straxen
import numpy as np
from warnings import warn
export, __all__ = strax.exporter()

from straxen.plugins.peaks.peak_positions_lpf import lpf_execute

@export
class EventS2PositionLPF(strax.Plugin):
    """
    LinPosFit pluging for S2 position reconstruction at event level
    """
    __version__ = '0.0.0'
    depends_on = ('event_area_per_channel', 'event_basics')
    provides = "event_s2_position_lpf"

    algorithm = 'lpf'
    compressor = 'zstd'
    parallel = True  # can set to "process" after #82

    min_reconstruction_area = straxen.URLConfig(
        help='Skip reconstruction if area (PE) is less than this',
        default=10, infer_type=False, )
    n_top_pmts = straxen.URLConfig(
        default=straxen.n_top_pmts, infer_type=False,
        help="Number of top PMTs")
    pmt_to_lxe = straxen.URLConfig(default=7, infer_type=False,
                                   help="Distance between the PMTs and the liquid interface")


    def infer_dtype(self):
        dtype = []
        for ptype in ['', 'alt_']:
            for xy in ['x', 'y']:
                dtype += [(f'event_{ptype}s2_{xy}_' + self.algorithm, np.float32,
                        f'Reconstructed {self.algorithm} {ptype}S2 {xy} position (cm), uncorrected'),
                        (f'event_{ptype}s2_err_{xy}_' + self.algorithm, np.float32,
                        f'Error on {self.algorithm} {ptype}S2 {xy} position (cm), 95p confidence interval'),
                        ]

            dtype += [(f'event_{ptype}s2_lpf_logl',np.float32, f'LinPosFit {ptype}S2 LogLikelihood value'),
                     (f'event_{ptype}s2_lpf_nuv',np.float32, f'LinPosFit {ptype}S2 parameter n, number of photons generated'),
                    (f'event_{ptype}s2_lpf_gamma', np.float32, f'LinPosFit {ptype}S2 gamma parameter, reflection'),
                    (f'event_{ptype}s2_lpf_n_in_fit',np.float32, f'LinPosFit {ptype}S2 number of PMTs included in the fit'),
                    (f'event_{ptype}s2_lpf_r0',np.float32, f'LinPosFit {ptype}S2 r0 parameter, reduced rate'), 
                    ]

        dtype += strax.time_fields
        return dtype

    def setup(self,):
        inch = 2.54 # cm
        pmt_pos = straxen.pmt_positions()
        self.pmt_pos = list(zip(pmt_pos['x'].values,pmt_pos['y'].values,np.repeat(self.pmt_to_lxe, self.n_top_pmts)))
        self.pmt_surface=(3*inch)**2*np.pi/4.


    def compute(self, events):

        result = np.ones(len(events), dtype=self.dtype)
        result['time'], result['endtime'] = events['time'], strax.endtime(events)

        for p_type in ['s2', 'alt_s2']:
            result[f'event_{p_type}_x_lpf'] *= float('nan')
            result[f'event_{p_type}_err_x_lpf'] *= float('nan')
            result[f'event_{p_type}_y_lpf'] *= float('nan')
            result[f'event_{p_type}_err_y_lpf'] *= float('nan')
            result[f'event_{p_type}_lpf_logl'] *= float('nan')
            result[f'event_{p_type}_lpf_nuv'] *= float('nan')
            result[f'event_{p_type}_lpf_gamma'] *= float('nan')
            result[f'event_{p_type}_lpf_n_in_fit'] *= float('nan')
            result[f'event_{p_type}_lpf_r0'] *= float('nan')


        for ix, ev in enumerate(events):
            for p_type in ['s2', 'alt_s2']:            
                if ev[p_type+'_area'] > self.min_reconstruction_area:

                    _top_pattern = ev[p_type + '_area_per_channel'][0:self.n_top_pmts]

                    try:

                        # Execute linearised position reconstruction fit
                        fit_result, xiter, used_hits = lpf_execute(self.pmt_pos[:self.n_top_pmts],
                                                                _top_pattern,
                                                                self.pmt_surface)

                        result[ix][f'event_{p_type}_x_lpf']     = fit_result[0]
                        result[ix][f'event_{p_type}_y_lpf']     = fit_result[1]
                        result[ix][f'event_{p_type}_lpf_nuv']  = fit_result[2]
                        result[ix][f'event_{p_type}_lpf_logl']  = fit_result[3]
                        result[ix][f'event_{p_type}_lpf_n_in_fit'] = fit_result[4]
                        result[ix][f'event_{p_type}_lpf_gamma'] = fit_result[5]
                        result[ix][f'event_{p_type}_lpf_r0']    = fit_result[6]
                        result[ix][f'event_{p_type}_err_x_lpf'] = fit_result[7]
                        result[ix][f'event_{p_type}_err_y_lpf'] = fit_result[8]

                    except Exception as e:
                        # Sometimes inverting the matrix fails.. 
                        # Let's just leave it as a NaN
                        print(e)
                        pass
            
        return result




