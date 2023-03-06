
from immutabledict import immutabledict
import numba
import numpy as np

import strax
import straxen



class AqmonHitsSV(straxen.AqmonHits):
    depends_on = ['raw_records_aqmon_sv']
    provides = ['aqmon_hits_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i2'), (('Channel/PMT number', 'channel'), '<i2'), (('Integral [ADC x samples]', 'area'), '<f4'), (('Index of sample in record in which hit starts', 'left'), '<i2'), (('Index of first sample in record just beyond hit (exclusive bound)', 'right'), '<i2'), (('For lone hits, index of sample in record where integration starts', 'left_integration'), '<i2'), (('For lone hits, index of first sample beyond integration region', 'right_integration'), '<i2'), (('Internal (temporary) index of fragment in which hit was found', 'record_i'), '<i4'), (('ADC threshold applied in order to find hits', 'threshold'), '<f4'), (('Maximum amplitude above baseline [ADC counts]', 'height'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class BayesPeakClassificationSV(straxen.BayesPeakClassification):
    depends_on = ['peaks_sv']
    provides = ['peak_classification_bayes_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('S1 ln probability', 'ln_prob_s1'), '<f4'), (('S2 ln probability', 'ln_prob_s2'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class CorrectedAreasSV(straxen.CorrectedAreas):
    depends_on = ['event_basics_sv', 'event_positions_sv']
    provides = ['corrected_areas_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Corrected area of main S1 [PE]', 'cs1'), '<f4'), (('Corrected area of main S1 [PE] before time-dep LY correction', 'cs1_wo_timecorr'), '<f4'), (('Corrected area of main S2 before elife correction (s2 xy correction + SEG/EE correction applied) [PE]', 'cs2_wo_elifecorr'), '<f4'), (('Corrected area of main S2 before SEG/EE and elife corrections(s2 xy correction applied) [PE]', 'cs2_wo_timecorr'), '<f4'), (('Fraction of area seen by the top PMT array for corrected main S2', 'cs2_area_fraction_top'), '<f4'), (('Corrected area of main S2 in the bottom PMT array [PE]', 'cs2_bottom'), '<f4'), (('Corrected area of main S2 [PE]', 'cs2'), '<f4'), (('Corrected area of alternate S1 [PE]', 'alt_cs1'), '<f4'), (('Corrected area of alternate S1 [PE] before time-dep LY correction', 'alt_cs1_wo_timecorr'), '<f4'), (('Corrected area of alternate S2 before elife correction (s2 xy correction + SEG/EE correction applied) [PE]', 'alt_cs2_wo_elifecorr'), '<f4'), (('Corrected area of alternate S2 before SEG/EE and elife corrections(s2 xy correction applied) [PE]', 'alt_cs2_wo_timecorr'), '<f4'), (('Fraction of area seen by the top PMT array for corrected alternate S2', 'alt_cs2_area_fraction_top'), '<f4'), (('Corrected area of alternate S2 in the bottom PMT array [PE]', 'alt_cs2_bottom'), '<f4'), (('Corrected area of alternate S2 [PE]', 'alt_cs2'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class DetectorSynchronizationSV(straxen.DetectorSynchronization):
    depends_on = ['raw_records_aqmon_sv', 'raw_records_aqmon_nv_sv', 'raw_records_aux_mv_sv']
    provides = ['detector_time_offsets_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Time offset for nV to synchronize with TPC in [ns]', 'time_offset_nv'), '<i8'), (('Time offset for mV to synchronize with TPC in [ns]', 'time_offset_mv'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class DistinctChannelsSV(straxen.DistinctChannels):
    depends_on = ['event_basics_sv', 'peaks_sv']
    provides = ['distinct_channels_sv']
    dtype = [(('Number of PMTs contributing to the secondary S1 that do not contribute to the main S1', 'alt_s1_distinct_channels'), '<i4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class EnergyEstimatesSV(straxen.EnergyEstimates):
    depends_on = ['corrected_areas_sv']
    provides = ['energy_estimates_sv']
    dtype = [(('Energy in light signal [keVee]', 'e_light'), '<f4'), (('Energy in charge signal [keVee]', 'e_charge'), '<f4'), (('Energy estimate [keVee]', 'e_ces'), '<f4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class EventAmbienceSV(straxen.EventAmbience):
    depends_on = ['event_basics_sv', 'peak_basics_sv', 'peak_ambience_sv']
    provides = ['event_ambience_sv']
    dtype = [(('Number of  lh before main S1', 's1_n_lh_before'), '<i2'), (('Number of  lh before main S2', 's2_n_lh_before'), '<i2'), (('Number of  s0 before main S1', 's1_n_s0_before'), '<i2'), (('Number of  s0 before main S2', 's2_n_s0_before'), '<i2'), (('Number of  s1 before main S1', 's1_n_s1_before'), '<i2'), (('Number of  s1 before main S2', 's2_n_s1_before'), '<i2'), (('Number of  s2 before main S1', 's1_n_s2_before'), '<i2'), (('Number of  s2 before main S2', 's2_n_s2_before'), '<i2'), (('Number of  s2 near main S1', 's1_n_s2_near'), '<i2'), (('Number of  s2 near main S2', 's2_n_s2_near'), '<i2'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class EventAreaPerChannelSV(straxen.EventAreaPerChannel):
    depends_on = ['event_basics_sv', 'peaks_sv']
    provides = ['event_area_per_channel_sv']
    dtype = [(('Area per channel for main S1', 's1_area_per_channel'), '<f4', (494,)), (('Waveform for main S1 [ PE / sample ]', 's1_data'), '<f4', (200,)), (('Top waveform for main S1 [ PE / sample ]', 's1_data_top'), '<f4', (200,)), (('Length of the interval in samples for main S1', 's1_length'), '<i4'), (('Width of one sample for main S1 [ns]', 's1_dt'), '<i4'), (('Area per channel for main S2', 's2_area_per_channel'), '<f4', (494,)), (('Waveform for main S2 [ PE / sample ]', 's2_data'), '<f4', (200,)), (('Top waveform for main S2 [ PE / sample ]', 's2_data_top'), '<f4', (200,)), (('Length of the interval in samples for main S2', 's2_length'), '<i4'), (('Width of one sample for main S2 [ns]', 's2_dt'), '<i4'), (('Area per channel for alternative S1', 'alt_s1_area_per_channel'), '<f4', (494,)), (('Waveform for alternative S1 [ PE / sample ]', 'alt_s1_data'), '<f4', (200,)), (('Top waveform for alternative S1 [ PE / sample ]', 'alt_s1_data_top'), '<f4', (200,)), (('Length of the interval in samples for alternative S1', 'alt_s1_length'), '<i4'), (('Width of one sample for alternative S1 [ns]', 'alt_s1_dt'), '<i4'), (('Area per channel for alternative S2', 'alt_s2_area_per_channel'), '<f4', (494,)), (('Waveform for alternative S2 [ PE / sample ]', 'alt_s2_data'), '<f4', (200,)), (('Top waveform for alternative S2 [ PE / sample ]', 'alt_s2_data_top'), '<f4', (200,)), (('Length of the interval in samples for alternative S2', 'alt_s2_length'), '<i4'), (('Width of one sample for alternative S2 [ns]', 'alt_s2_dt'), '<i4'), (('Main S1 count of contributing PMTs', 's1_n_channels'), '<i2'), (('Main S1 top count of contributing PMTs', 's1_top_n_channels'), '<i2'), (('Main S1 bottom count of contributing PMTs', 's1_bottom_n_channels'), '<i2'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class EventBasicsSV(straxen.EventBasics):
    depends_on = ['events_sv', 'peak_basics_sv', 'peak_positions_sv', 'peak_proximity_sv']
    provides = ['event_basics_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Number of peaks in the event', 'n_peaks'), '<i4'), (('Drift time between main S1 and S2 in ns', 'drift_time'), '<f4'), (('Event number in this dataset', 'event_number'), '<i8'), (('Main S1 peak index in event', 's1_index'), '<i4'), (('Alternate S1 peak index in event', 'alt_s1_index'), '<i4'), (('Main S1 start time since unix epoch [ns]', 's1_time'), '<i8'), (('Alternate S1 start time since unix epoch [ns]', 'alt_s1_time'), '<i8'), (('Main S1 weighted center time since unix epoch [ns]', 's1_center_time'), '<i8'), (('Alternate S1 weighted center time since unix epoch [ns]', 'alt_s1_center_time'), '<i8'), (('Main S1 end time since unix epoch [ns]', 's1_endtime'), '<i8'), (('Alternate S1 end time since unix epoch [ns]', 'alt_s1_endtime'), '<i8'), (('Main S1 area, uncorrected [PE]', 's1_area'), '<f4'), (('Alternate S1 area, uncorrected [PE]', 'alt_s1_area'), '<f4'), (('Main S1 count of contributing PMTs', 's1_n_channels'), '<i2'), (('Alternate S1 count of contributing PMTs', 'alt_s1_n_channels'), '<i2'), (('Main S1 count of hits contributing at least one sample to the peak', 's1_n_hits'), '<i2'), (('Alternate S1 count of hits contributing at least one sample to the peak', 'alt_s1_n_hits'), '<i2'), (('Main S1 number of competing peaks', 's1_n_competing'), '<i4'), (('Alternate S1 number of competing peaks', 'alt_s1_n_competing'), '<i4'), (('Main S1 PMT number which contributes the most PE', 's1_max_pmt'), '<i2'), (('Alternate S1 PMT number which contributes the most PE', 'alt_s1_max_pmt'), '<i2'), (('Main S1 area in the largest-contributing PMT (PE)', 's1_max_pmt_area'), '<f4'), (('Alternate S1 area in the largest-contributing PMT (PE)', 'alt_s1_max_pmt_area'), '<f4'), (('Main S1 width, 50% area [ns]', 's1_range_50p_area'), '<f4'), (('Alternate S1 width, 50% area [ns]', 'alt_s1_range_50p_area'), '<f4'), (('Main S1 width, 90% area [ns]', 's1_range_90p_area'), '<f4'), (('Alternate S1 width, 90% area [ns]', 'alt_s1_range_90p_area'), '<f4'), (('Main S1 time between 10% and 50% area quantiles [ns]', 's1_rise_time'), '<f4'), (('Alternate S1 time between 10% and 50% area quantiles [ns]', 'alt_s1_rise_time'), '<f4'), (('Main S1 fraction of area seen by the top PMT array', 's1_area_fraction_top'), '<f4'), (('Alternate S1 fraction of area seen by the top PMT array', 'alt_s1_area_fraction_top'), '<f4'), (('Main S1 Channel within tight range of mean', 's1_tight_coincidence'), '<i2'), (('Alternate S1 Channel within tight range of mean', 'alt_s1_tight_coincidence'), '<i2'), (('Main S1 Total number of saturated channels', 's1_n_saturated_channels'), '<i2'), (('Alternate S1 Total number of saturated channels', 'alt_s1_n_saturated_channels'), '<i2'), (('Drift time using alternate S1 [ns]', 'alt_s1_interaction_drift_time'), '<f4'), (('Time between main and alternate S1 [ns]', 'alt_s1_delay'), '<i4'), (('Main S2 peak index in event', 's2_index'), '<i4'), (('Alternate S2 peak index in event', 'alt_s2_index'), '<i4'), (('Main S2 start time since unix epoch [ns]', 's2_time'), '<i8'), (('Alternate S2 start time since unix epoch [ns]', 'alt_s2_time'), '<i8'), (('Main S2 weighted center time since unix epoch [ns]', 's2_center_time'), '<i8'), (('Alternate S2 weighted center time since unix epoch [ns]', 'alt_s2_center_time'), '<i8'), (('Main S2 end time since unix epoch [ns]', 's2_endtime'), '<i8'), (('Alternate S2 end time since unix epoch [ns]', 'alt_s2_endtime'), '<i8'), (('Main S2 area, uncorrected [PE]', 's2_area'), '<f4'), (('Alternate S2 area, uncorrected [PE]', 'alt_s2_area'), '<f4'), (('Main S2 count of contributing PMTs', 's2_n_channels'), '<i2'), (('Alternate S2 count of contributing PMTs', 'alt_s2_n_channels'), '<i2'), (('Main S2 count of hits contributing at least one sample to the peak', 's2_n_hits'), '<i2'), (('Alternate S2 count of hits contributing at least one sample to the peak', 'alt_s2_n_hits'), '<i2'), (('Main S2 number of competing peaks', 's2_n_competing'), '<i4'), (('Alternate S2 number of competing peaks', 'alt_s2_n_competing'), '<i4'), (('Main S2 PMT number which contributes the most PE', 's2_max_pmt'), '<i2'), (('Alternate S2 PMT number which contributes the most PE', 'alt_s2_max_pmt'), '<i2'), (('Main S2 area in the largest-contributing PMT (PE)', 's2_max_pmt_area'), '<f4'), (('Alternate S2 area in the largest-contributing PMT (PE)', 'alt_s2_max_pmt_area'), '<f4'), (('Main S2 width, 50% area [ns]', 's2_range_50p_area'), '<f4'), (('Alternate S2 width, 50% area [ns]', 'alt_s2_range_50p_area'), '<f4'), (('Main S2 width, 90% area [ns]', 's2_range_90p_area'), '<f4'), (('Alternate S2 width, 90% area [ns]', 'alt_s2_range_90p_area'), '<f4'), (('Main S2 time between 10% and 50% area quantiles [ns]', 's2_rise_time'), '<f4'), (('Alternate S2 time between 10% and 50% area quantiles [ns]', 'alt_s2_rise_time'), '<f4'), (('Main S2 fraction of area seen by the top PMT array', 's2_area_fraction_top'), '<f4'), (('Alternate S2 fraction of area seen by the top PMT array', 'alt_s2_area_fraction_top'), '<f4'), (('Main S2 Channel within tight range of mean', 's2_tight_coincidence'), '<i2'), (('Alternate S2 Channel within tight range of mean', 'alt_s2_tight_coincidence'), '<i2'), (('Main S2 Total number of saturated channels', 's2_n_saturated_channels'), '<i2'), (('Alternate S2 Total number of saturated channels', 'alt_s2_n_saturated_channels'), '<i2'), (('Drift time using alternate S2 [ns]', 'alt_s2_interaction_drift_time'), '<f4'), (('Time between main and alternate S2 [ns]', 'alt_s2_delay'), '<i4'), (('Main S2 reconstructed X position, uncorrected [cm]', 's2_x'), '<f4'), (('Main S2 reconstructed Y position, uncorrected [cm]', 's2_y'), '<f4'), (('Alternate S2 reconstructed X position, uncorrected [cm]', 'alt_s2_x'), '<f4'), (('Alternate S2 reconstructed Y position, uncorrected [cm]', 'alt_s2_y'), '<f4'), (('Sum of areas before Main S2 [PE]', 'area_before_main_s2'), '<f4'), (('The largest S2 before the Main S2 [PE]', 'large_s2_before_main_s2'), '<f4'), (('Main S2 cnn-reconstructed X position, uncorrected [cm]', 's2_x_cnn'), '<f4'), (('Main S2 cnn-reconstructed Y position, uncorrected [cm]', 's2_y_cnn'), '<f4'), (('Alternate S2 cnn-reconstructed X position, uncorrected [cm]', 'alt_s2_x_cnn'), '<f4'), (('Alternate S2 cnn-reconstructed Y position, uncorrected [cm]', 'alt_s2_y_cnn'), '<f4'), (('Main S2 gcn-reconstructed X position, uncorrected [cm]', 's2_x_gcn'), '<f4'), (('Main S2 gcn-reconstructed Y position, uncorrected [cm]', 's2_y_gcn'), '<f4'), (('Alternate S2 gcn-reconstructed X position, uncorrected [cm]', 'alt_s2_x_gcn'), '<f4'), (('Alternate S2 gcn-reconstructed Y position, uncorrected [cm]', 'alt_s2_y_gcn'), '<f4'), (('Main S2 mlp-reconstructed X position, uncorrected [cm]', 's2_x_mlp'), '<f4'), (('Main S2 mlp-reconstructed Y position, uncorrected [cm]', 's2_y_mlp'), '<f4'), (('Alternate S2 mlp-reconstructed X position, uncorrected [cm]', 'alt_s2_x_mlp'), '<f4'), (('Alternate S2 mlp-reconstructed Y position, uncorrected [cm]', 'alt_s2_y_mlp'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class EventInfoSV(straxen.EventInfo):
    depends_on = ['event_basics_sv', 'event_positions_sv', 'corrected_areas_sv', 'energy_estimates_sv']
    provides = ['event_info_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Corrected area of main S1 [PE]', 'cs1'), '<f4'), (('Corrected area of main S1 [PE] before time-dep LY correction', 'cs1_wo_timecorr'), '<f4'), (('Corrected area of main S2 before elife correction (s2 xy correction + SEG/EE correction applied) [PE]', 'cs2_wo_elifecorr'), '<f4'), (('Corrected area of main S2 before SEG/EE and elife corrections(s2 xy correction applied) [PE]', 'cs2_wo_timecorr'), '<f4'), (('Fraction of area seen by the top PMT array for corrected main S2', 'cs2_area_fraction_top'), '<f4'), (('Corrected area of main S2 in the bottom PMT array [PE]', 'cs2_bottom'), '<f4'), (('Corrected area of main S2 [PE]', 'cs2'), '<f4'), (('Corrected area of alternate S1 [PE]', 'alt_cs1'), '<f4'), (('Corrected area of alternate S1 [PE] before time-dep LY correction', 'alt_cs1_wo_timecorr'), '<f4'), (('Corrected area of alternate S2 before elife correction (s2 xy correction + SEG/EE correction applied) [PE]', 'alt_cs2_wo_elifecorr'), '<f4'), (('Corrected area of alternate S2 before SEG/EE and elife corrections(s2 xy correction applied) [PE]', 'alt_cs2_wo_timecorr'), '<f4'), (('Fraction of area seen by the top PMT array for corrected alternate S2', 'alt_cs2_area_fraction_top'), '<f4'), (('Corrected area of alternate S2 in the bottom PMT array [PE]', 'alt_cs2_bottom'), '<f4'), (('Corrected area of alternate S2 [PE]', 'alt_cs2'), '<f4'), (('Energy in light signal [keVee]', 'e_light'), '<f4'), (('Energy in charge signal [keVee]', 'e_charge'), '<f4'), (('Energy estimate [keVee]', 'e_ces'), '<f4'), (('Number of peaks in the event', 'n_peaks'), '<i4'), (('Drift time between main S1 and S2 in ns', 'drift_time'), '<f4'), (('Event number in this dataset', 'event_number'), '<i8'), (('Main S1 peak index in event', 's1_index'), '<i4'), (('Alternate S1 peak index in event', 'alt_s1_index'), '<i4'), (('Main S1 start time since unix epoch [ns]', 's1_time'), '<i8'), (('Alternate S1 start time since unix epoch [ns]', 'alt_s1_time'), '<i8'), (('Main S1 weighted center time since unix epoch [ns]', 's1_center_time'), '<i8'), (('Alternate S1 weighted center time since unix epoch [ns]', 'alt_s1_center_time'), '<i8'), (('Main S1 end time since unix epoch [ns]', 's1_endtime'), '<i8'), (('Alternate S1 end time since unix epoch [ns]', 'alt_s1_endtime'), '<i8'), (('Main S1 area, uncorrected [PE]', 's1_area'), '<f4'), (('Alternate S1 area, uncorrected [PE]', 'alt_s1_area'), '<f4'), (('Main S1 count of contributing PMTs', 's1_n_channels'), '<i2'), (('Alternate S1 count of contributing PMTs', 'alt_s1_n_channels'), '<i2'), (('Main S1 count of hits contributing at least one sample to the peak', 's1_n_hits'), '<i2'), (('Alternate S1 count of hits contributing at least one sample to the peak', 'alt_s1_n_hits'), '<i2'), (('Main S1 number of competing peaks', 's1_n_competing'), '<i4'), (('Alternate S1 number of competing peaks', 'alt_s1_n_competing'), '<i4'), (('Main S1 PMT number which contributes the most PE', 's1_max_pmt'), '<i2'), (('Alternate S1 PMT number which contributes the most PE', 'alt_s1_max_pmt'), '<i2'), (('Main S1 area in the largest-contributing PMT (PE)', 's1_max_pmt_area'), '<f4'), (('Alternate S1 area in the largest-contributing PMT (PE)', 'alt_s1_max_pmt_area'), '<f4'), (('Main S1 width, 50% area [ns]', 's1_range_50p_area'), '<f4'), (('Alternate S1 width, 50% area [ns]', 'alt_s1_range_50p_area'), '<f4'), (('Main S1 width, 90% area [ns]', 's1_range_90p_area'), '<f4'), (('Alternate S1 width, 90% area [ns]', 'alt_s1_range_90p_area'), '<f4'), (('Main S1 time between 10% and 50% area quantiles [ns]', 's1_rise_time'), '<f4'), (('Alternate S1 time between 10% and 50% area quantiles [ns]', 'alt_s1_rise_time'), '<f4'), (('Main S1 fraction of area seen by the top PMT array', 's1_area_fraction_top'), '<f4'), (('Alternate S1 fraction of area seen by the top PMT array', 'alt_s1_area_fraction_top'), '<f4'), (('Main S1 Channel within tight range of mean', 's1_tight_coincidence'), '<i2'), (('Alternate S1 Channel within tight range of mean', 'alt_s1_tight_coincidence'), '<i2'), (('Main S1 Total number of saturated channels', 's1_n_saturated_channels'), '<i2'), (('Alternate S1 Total number of saturated channels', 'alt_s1_n_saturated_channels'), '<i2'), (('Drift time using alternate S1 [ns]', 'alt_s1_interaction_drift_time'), '<f4'), (('Time between main and alternate S1 [ns]', 'alt_s1_delay'), '<i4'), (('Main S2 peak index in event', 's2_index'), '<i4'), (('Alternate S2 peak index in event', 'alt_s2_index'), '<i4'), (('Main S2 start time since unix epoch [ns]', 's2_time'), '<i8'), (('Alternate S2 start time since unix epoch [ns]', 'alt_s2_time'), '<i8'), (('Main S2 weighted center time since unix epoch [ns]', 's2_center_time'), '<i8'), (('Alternate S2 weighted center time since unix epoch [ns]', 'alt_s2_center_time'), '<i8'), (('Main S2 end time since unix epoch [ns]', 's2_endtime'), '<i8'), (('Alternate S2 end time since unix epoch [ns]', 'alt_s2_endtime'), '<i8'), (('Main S2 area, uncorrected [PE]', 's2_area'), '<f4'), (('Alternate S2 area, uncorrected [PE]', 'alt_s2_area'), '<f4'), (('Main S2 count of contributing PMTs', 's2_n_channels'), '<i2'), (('Alternate S2 count of contributing PMTs', 'alt_s2_n_channels'), '<i2'), (('Main S2 count of hits contributing at least one sample to the peak', 's2_n_hits'), '<i2'), (('Alternate S2 count of hits contributing at least one sample to the peak', 'alt_s2_n_hits'), '<i2'), (('Main S2 number of competing peaks', 's2_n_competing'), '<i4'), (('Alternate S2 number of competing peaks', 'alt_s2_n_competing'), '<i4'), (('Main S2 PMT number which contributes the most PE', 's2_max_pmt'), '<i2'), (('Alternate S2 PMT number which contributes the most PE', 'alt_s2_max_pmt'), '<i2'), (('Main S2 area in the largest-contributing PMT (PE)', 's2_max_pmt_area'), '<f4'), (('Alternate S2 area in the largest-contributing PMT (PE)', 'alt_s2_max_pmt_area'), '<f4'), (('Main S2 width, 50% area [ns]', 's2_range_50p_area'), '<f4'), (('Alternate S2 width, 50% area [ns]', 'alt_s2_range_50p_area'), '<f4'), (('Main S2 width, 90% area [ns]', 's2_range_90p_area'), '<f4'), (('Alternate S2 width, 90% area [ns]', 'alt_s2_range_90p_area'), '<f4'), (('Main S2 time between 10% and 50% area quantiles [ns]', 's2_rise_time'), '<f4'), (('Alternate S2 time between 10% and 50% area quantiles [ns]', 'alt_s2_rise_time'), '<f4'), (('Main S2 fraction of area seen by the top PMT array', 's2_area_fraction_top'), '<f4'), (('Alternate S2 fraction of area seen by the top PMT array', 'alt_s2_area_fraction_top'), '<f4'), (('Main S2 Channel within tight range of mean', 's2_tight_coincidence'), '<i2'), (('Alternate S2 Channel within tight range of mean', 'alt_s2_tight_coincidence'), '<i2'), (('Main S2 Total number of saturated channels', 's2_n_saturated_channels'), '<i2'), (('Alternate S2 Total number of saturated channels', 'alt_s2_n_saturated_channels'), '<i2'), (('Drift time using alternate S2 [ns]', 'alt_s2_interaction_drift_time'), '<f4'), (('Time between main and alternate S2 [ns]', 'alt_s2_delay'), '<i4'), (('Main S2 reconstructed X position, uncorrected [cm]', 's2_x'), '<f4'), (('Main S2 reconstructed Y position, uncorrected [cm]', 's2_y'), '<f4'), (('Alternate S2 reconstructed X position, uncorrected [cm]', 'alt_s2_x'), '<f4'), (('Alternate S2 reconstructed Y position, uncorrected [cm]', 'alt_s2_y'), '<f4'), (('Sum of areas before Main S2 [PE]', 'area_before_main_s2'), '<f4'), (('The largest S2 before the Main S2 [PE]', 'large_s2_before_main_s2'), '<f4'), (('Main S2 cnn-reconstructed X position, uncorrected [cm]', 's2_x_cnn'), '<f4'), (('Main S2 cnn-reconstructed Y position, uncorrected [cm]', 's2_y_cnn'), '<f4'), (('Alternate S2 cnn-reconstructed X position, uncorrected [cm]', 'alt_s2_x_cnn'), '<f4'), (('Alternate S2 cnn-reconstructed Y position, uncorrected [cm]', 'alt_s2_y_cnn'), '<f4'), (('Main S2 gcn-reconstructed X position, uncorrected [cm]', 's2_x_gcn'), '<f4'), (('Main S2 gcn-reconstructed Y position, uncorrected [cm]', 's2_y_gcn'), '<f4'), (('Alternate S2 gcn-reconstructed X position, uncorrected [cm]', 'alt_s2_x_gcn'), '<f4'), (('Alternate S2 gcn-reconstructed Y position, uncorrected [cm]', 'alt_s2_y_gcn'), '<f4'), (('Main S2 mlp-reconstructed X position, uncorrected [cm]', 's2_x_mlp'), '<f4'), (('Main S2 mlp-reconstructed Y position, uncorrected [cm]', 's2_y_mlp'), '<f4'), (('Alternate S2 mlp-reconstructed X position, uncorrected [cm]', 'alt_s2_x_mlp'), '<f4'), (('Alternate S2 mlp-reconstructed Y position, uncorrected [cm]', 'alt_s2_y_mlp'), '<f4'), (('Main interaction x-position, field-distortion corrected (cm)', 'x'), '<f4'), (('Alternative S1 interaction (rel. main S2) x-position, field-distortion corrected (cm)', 'alt_s1_x_fdc'), '<f4'), (('Alternative S2 interaction (rel. main S1) x-position, field-distortion corrected (cm)', 'alt_s2_x_fdc'), '<f4'), (('Main interaction y-position, field-distortion corrected (cm)', 'y'), '<f4'), (('Alternative S1 interaction (rel. main S2) y-position, field-distortion corrected (cm)', 'alt_s1_y_fdc'), '<f4'), (('Alternative S2 interaction (rel. main S1) y-position, field-distortion corrected (cm)', 'alt_s2_y_fdc'), '<f4'), (('Main interaction r-position, field-distortion corrected (cm)', 'r'), '<f4'), (('Alternative S1 interaction (rel. main S2) r-position, field-distortion corrected (cm)', 'alt_s1_r_fdc'), '<f4'), (('Alternative S2 interaction (rel. main S1) r-position, field-distortion corrected (cm)', 'alt_s2_r_fdc'), '<f4'), (('Interaction z-position, using mean drift velocity only (cm)', 'z'), '<f4'), (('Alternative S1 z-position (rel. main S2), using mean drift velocity only (cm)', 'alt_s1_z'), '<f4'), (('Alternative S2 z-position (rel. main S1), using mean drift velocity only (cm)', 'alt_s2_z'), '<f4'), (('Main interaction r-position with observed position (cm)', 'r_naive'), '<f4'), (('Alternative S1 interaction (rel. main S2) r-position with observed position (cm)', 'alt_s1_r_naive'), '<f4'), (('Alternative S2 interaction (rel. main S1) r-position with observed position (cm)', 'alt_s2_r_naive'), '<f4'), (('Main interaction z-position with observed position (cm)', 'z_naive'), '<f4'), (('Alternative S1 interaction (rel. main S2) z-position with observed position (cm)', 'alt_s1_z_naive'), '<f4'), (('Alternative S2 interaction (rel. main S1) z-position with observed position (cm)', 'alt_s2_z_naive'), '<f4'), (('Correction added to r_naive for field distortion (cm)', 'r_field_distortion_correction'), '<f4'), (('Correction added to alt_s1_r_naive for field distortion (cm)', 'alt_s1_r_field_distortion_correction'), '<f4'), (('Correction added to alt_s2_r_naive for field distortion (cm)', 'alt_s2_r_field_distortion_correction'), '<f4'), (('Correction added to z_naive for field distortion (cm)', 'z_field_distortion_correction'), '<f4'), (('Correction added to alt_s1_z_naive for field distortion (cm)', 'alt_s1_z_field_distortion_correction'), '<f4'), (('Correction added to alt_s2_z_naive for field distortion (cm)', 'alt_s2_z_field_distortion_correction'), '<f4'), (('Alternative S1 (rel. main S2) interaction angular position (radians)', 'alt_s1_theta'), '<f4'), (('Alternative S2 (rel. main S1) interaction angular position (radians)', 'alt_s2_theta'), '<f4'), (('Main interaction angular position (radians)', 'theta'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class EventInfoDoubleSV(straxen.EventInfoDouble):
    depends_on = ['event_info_sv', 'distinct_channels_sv']
    provides = ['event_info_double_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Corrected area of main S1 [PE]', 'cs1_a'), '<f4'), (('Corrected area of main S1 [PE] before time-dep LY correction', 'cs1_a_wo_timecorr'), '<f4'), (('Corrected area of main S2 before elife correction (s2 xy correction + SEG/EE correction applied) [PE]', 'cs2_a_wo_elifecorr'), '<f4'), (('Corrected area of main S2 before SEG/EE and elife corrections(s2 xy correction applied) [PE]', 'cs2_a_wo_timecorr'), '<f4'), (('Fraction of area seen by the top PMT array for corrected main S2', 'cs2_a_area_fraction_top'), '<f4'), (('Corrected area of main S2 in the bottom PMT array [PE]', 'cs2_a_bottom'), '<f4'), (('Corrected area of main S2 [PE]', 'cs2_a'), '<f4'), (('Corrected area of alternate S1 [PE]', 'cs1_b'), '<f4'), (('Corrected area of alternate S1 [PE] before time-dep LY correction', 'cs1_b_wo_timecorr'), '<f4'), (('Corrected area of alternate S2 before elife correction (s2 xy correction + SEG/EE correction applied) [PE]', 'cs2_b_wo_elifecorr'), '<f4'), (('Corrected area of alternate S2 before SEG/EE and elife corrections(s2 xy correction applied) [PE]', 'cs2_b_wo_timecorr'), '<f4'), (('Fraction of area seen by the top PMT array for corrected alternate S2', 'cs2_b_area_fraction_top'), '<f4'), (('Corrected area of alternate S2 in the bottom PMT array [PE]', 'cs2_b_bottom'), '<f4'), (('Corrected area of alternate S2 [PE]', 'cs2_b'), '<f4'), (('Energy in light signal [keVee]', 'e_light'), '<f4'), (('Energy in charge signal [keVee]', 'e_charge'), '<f4'), (('Energy estimate [keVee]', 'e_ces'), '<f4'), (('Number of peaks in the event', 'n_peaks'), '<i4'), (('Drift time between main S1 and S2 in ns', 'drift_time'), '<f4'), (('Event number in this dataset', 'event_number'), '<i8'), (('Main S1 peak index in event', 's1_a_index'), '<i4'), (('Alternate S1 peak index in event', 's1_b_index'), '<i4'), (('Main S1 start time since unix epoch [ns]', 's1_a_time'), '<i8'), (('Alternate S1 start time since unix epoch [ns]', 's1_b_time'), '<i8'), (('Main S1 weighted center time since unix epoch [ns]', 's1_a_center_time'), '<i8'), (('Alternate S1 weighted center time since unix epoch [ns]', 's1_b_center_time'), '<i8'), (('Main S1 end time since unix epoch [ns]', 's1_a_endtime'), '<i8'), (('Alternate S1 end time since unix epoch [ns]', 's1_b_endtime'), '<i8'), (('Main S1 area, uncorrected [PE]', 's1_a_area'), '<f4'), (('Alternate S1 area, uncorrected [PE]', 's1_b_area'), '<f4'), (('Main S1 count of contributing PMTs', 's1_a_n_channels'), '<i2'), (('Alternate S1 count of contributing PMTs', 's1_b_n_channels'), '<i2'), (('Main S1 count of hits contributing at least one sample to the peak', 's1_a_n_hits'), '<i2'), (('Alternate S1 count of hits contributing at least one sample to the peak', 's1_b_n_hits'), '<i2'), (('Main S1 number of competing peaks', 's1_a_n_competing'), '<i4'), (('Alternate S1 number of competing peaks', 's1_b_n_competing'), '<i4'), (('Main S1 PMT number which contributes the most PE', 's1_a_max_pmt'), '<i2'), (('Alternate S1 PMT number which contributes the most PE', 's1_b_max_pmt'), '<i2'), (('Main S1 area in the largest-contributing PMT (PE)', 's1_a_max_pmt_area'), '<f4'), (('Alternate S1 area in the largest-contributing PMT (PE)', 's1_b_max_pmt_area'), '<f4'), (('Main S1 width, 50% area [ns]', 's1_a_range_50p_area'), '<f4'), (('Alternate S1 width, 50% area [ns]', 's1_b_range_50p_area'), '<f4'), (('Main S1 width, 90% area [ns]', 's1_a_range_90p_area'), '<f4'), (('Alternate S1 width, 90% area [ns]', 's1_b_range_90p_area'), '<f4'), (('Main S1 time between 10% and 50% area quantiles [ns]', 's1_a_rise_time'), '<f4'), (('Alternate S1 time between 10% and 50% area quantiles [ns]', 's1_b_rise_time'), '<f4'), (('Main S1 fraction of area seen by the top PMT array', 's1_a_area_fraction_top'), '<f4'), (('Alternate S1 fraction of area seen by the top PMT array', 's1_b_area_fraction_top'), '<f4'), (('Main S1 Channel within tight range of mean', 's1_a_tight_coincidence'), '<i2'), (('Alternate S1 Channel within tight range of mean', 's1_b_tight_coincidence'), '<i2'), (('Main S1 Total number of saturated channels', 's1_a_n_saturated_channels'), '<i2'), (('Alternate S1 Total number of saturated channels', 's1_b_n_saturated_channels'), '<i2'), (('Drift time using alternate S1 [ns]', 's1_b_interaction_drift_time'), '<f4'), (('Time between main and alternate S1 [ns]', 'ds_s1_dt'), '<i4'), (('Main S2 peak index in event', 's2_a_index'), '<i4'), (('Alternate S2 peak index in event', 's2_b_index'), '<i4'), (('Main S2 start time since unix epoch [ns]', 's2_a_time'), '<i8'), (('Alternate S2 start time since unix epoch [ns]', 's2_b_time'), '<i8'), (('Main S2 weighted center time since unix epoch [ns]', 's2_a_center_time'), '<i8'), (('Alternate S2 weighted center time since unix epoch [ns]', 's2_b_center_time'), '<i8'), (('Main S2 end time since unix epoch [ns]', 's2_a_endtime'), '<i8'), (('Alternate S2 end time since unix epoch [ns]', 's2_b_endtime'), '<i8'), (('Main S2 area, uncorrected [PE]', 's2_a_area'), '<f4'), (('Alternate S2 area, uncorrected [PE]', 's2_b_area'), '<f4'), (('Main S2 count of contributing PMTs', 's2_a_n_channels'), '<i2'), (('Alternate S2 count of contributing PMTs', 's2_b_n_channels'), '<i2'), (('Main S2 count of hits contributing at least one sample to the peak', 's2_a_n_hits'), '<i2'), (('Alternate S2 count of hits contributing at least one sample to the peak', 's2_b_n_hits'), '<i2'), (('Main S2 number of competing peaks', 's2_a_n_competing'), '<i4'), (('Alternate S2 number of competing peaks', 's2_b_n_competing'), '<i4'), (('Main S2 PMT number which contributes the most PE', 's2_a_max_pmt'), '<i2'), (('Alternate S2 PMT number which contributes the most PE', 's2_b_max_pmt'), '<i2'), (('Main S2 area in the largest-contributing PMT (PE)', 's2_a_max_pmt_area'), '<f4'), (('Alternate S2 area in the largest-contributing PMT (PE)', 's2_b_max_pmt_area'), '<f4'), (('Main S2 width, 50% area [ns]', 's2_a_range_50p_area'), '<f4'), (('Alternate S2 width, 50% area [ns]', 's2_b_range_50p_area'), '<f4'), (('Main S2 width, 90% area [ns]', 's2_a_range_90p_area'), '<f4'), (('Alternate S2 width, 90% area [ns]', 's2_b_range_90p_area'), '<f4'), (('Main S2 time between 10% and 50% area quantiles [ns]', 's2_a_rise_time'), '<f4'), (('Alternate S2 time between 10% and 50% area quantiles [ns]', 's2_b_rise_time'), '<f4'), (('Main S2 fraction of area seen by the top PMT array', 's2_a_area_fraction_top'), '<f4'), (('Alternate S2 fraction of area seen by the top PMT array', 's2_b_area_fraction_top'), '<f4'), (('Main S2 Channel within tight range of mean', 's2_a_tight_coincidence'), '<i2'), (('Alternate S2 Channel within tight range of mean', 's2_b_tight_coincidence'), '<i2'), (('Main S2 Total number of saturated channels', 's2_a_n_saturated_channels'), '<i2'), (('Alternate S2 Total number of saturated channels', 's2_b_n_saturated_channels'), '<i2'), (('Drift time using alternate S2 [ns]', 's2_b_interaction_drift_time'), '<f4'), (('Time between main and alternate S2 [ns]', 'ds_s2_dt'), '<i4'), (('Main S2 reconstructed X position, uncorrected [cm]', 's2_a_x'), '<f4'), (('Main S2 reconstructed Y position, uncorrected [cm]', 's2_a_y'), '<f4'), (('Alternate S2 reconstructed X position, uncorrected [cm]', 's2_b_x'), '<f4'), (('Alternate S2 reconstructed Y position, uncorrected [cm]', 's2_b_y'), '<f4'), (('Sum of areas before Main S2 [PE]', 'area_before_main_s2'), '<f4'), (('The largest S2 before the Main S2 [PE]', 'large_s2_before_main_s2'), '<f4'), (('Main S2 cnn-reconstructed X position, uncorrected [cm]', 's2_a_x_cnn'), '<f4'), (('Main S2 cnn-reconstructed Y position, uncorrected [cm]', 's2_a_y_cnn'), '<f4'), (('Alternate S2 cnn-reconstructed X position, uncorrected [cm]', 's2_b_x_cnn'), '<f4'), (('Alternate S2 cnn-reconstructed Y position, uncorrected [cm]', 's2_b_y_cnn'), '<f4'), (('Main S2 gcn-reconstructed X position, uncorrected [cm]', 's2_a_x_gcn'), '<f4'), (('Main S2 gcn-reconstructed Y position, uncorrected [cm]', 's2_a_y_gcn'), '<f4'), (('Alternate S2 gcn-reconstructed X position, uncorrected [cm]', 's2_b_x_gcn'), '<f4'), (('Alternate S2 gcn-reconstructed Y position, uncorrected [cm]', 's2_b_y_gcn'), '<f4'), (('Main S2 mlp-reconstructed X position, uncorrected [cm]', 's2_a_x_mlp'), '<f4'), (('Main S2 mlp-reconstructed Y position, uncorrected [cm]', 's2_a_y_mlp'), '<f4'), (('Alternate S2 mlp-reconstructed X position, uncorrected [cm]', 's2_b_x_mlp'), '<f4'), (('Alternate S2 mlp-reconstructed Y position, uncorrected [cm]', 's2_b_y_mlp'), '<f4'), (('Main interaction x-position, field-distortion corrected (cm)', 'x'), '<f4'), (('Alternative S1 interaction (rel. main S2) x-position, field-distortion corrected (cm)', 's1_b_x_fdc'), '<f4'), (('Alternative S2 interaction (rel. main S1) x-position, field-distortion corrected (cm)', 's2_b_x_fdc'), '<f4'), (('Main interaction y-position, field-distortion corrected (cm)', 'y'), '<f4'), (('Alternative S1 interaction (rel. main S2) y-position, field-distortion corrected (cm)', 's1_b_y_fdc'), '<f4'), (('Alternative S2 interaction (rel. main S1) y-position, field-distortion corrected (cm)', 's2_b_y_fdc'), '<f4'), (('Main interaction r-position, field-distortion corrected (cm)', 'r'), '<f4'), (('Alternative S1 interaction (rel. main S2) r-position, field-distortion corrected (cm)', 's1_b_r_fdc'), '<f4'), (('Alternative S2 interaction (rel. main S1) r-position, field-distortion corrected (cm)', 's2_b_r_fdc'), '<f4'), (('Interaction z-position, using mean drift velocity only (cm)', 'z'), '<f4'), (('Alternative S1 z-position (rel. main S2), using mean drift velocity only (cm)', 's1_b_z'), '<f4'), (('Alternative S2 z-position (rel. main S1), using mean drift velocity only (cm)', 's2_b_z'), '<f4'), (('Main interaction r-position with observed position (cm)', 'r_naive'), '<f4'), (('Alternative S1 interaction (rel. main S2) r-position with observed position (cm)', 's1_b_r_naive'), '<f4'), (('Alternative S2 interaction (rel. main S1) r-position with observed position (cm)', 's2_b_r_naive'), '<f4'), (('Main interaction z-position with observed position (cm)', 'z_naive'), '<f4'), (('Alternative S1 interaction (rel. main S2) z-position with observed position (cm)', 's1_b_z_naive'), '<f4'), (('Alternative S2 interaction (rel. main S1) z-position with observed position (cm)', 's2_b_z_naive'), '<f4'), (('Correction added to r_naive for field distortion (cm)', 'r_field_distortion_correction'), '<f4'), (('Correction added to alt_s1_r_naive for field distortion (cm)', 's1_b_r_field_distortion_correction'), '<f4'), (('Correction added to alt_s2_r_naive for field distortion (cm)', 's2_b_r_field_distortion_correction'), '<f4'), (('Correction added to z_naive for field distortion (cm)', 'z_field_distortion_correction'), '<f4'), (('Correction added to alt_s1_z_naive for field distortion (cm)', 's1_b_z_field_distortion_correction'), '<f4'), (('Correction added to alt_s2_z_naive for field distortion (cm)', 's2_b_z_field_distortion_correction'), '<f4'), (('Alternative S1 (rel. main S2) interaction angular position (radians)', 's1_b_theta'), '<f4'), (('Alternative S2 (rel. main S1) interaction angular position (radians)', 's2_b_theta'), '<f4'), (('Main interaction angular position (radians)', 'theta'), '<f4'), (('Number of PMTs contributing to the secondary S1 that do not contribute to the main S1', 's1_b_distinct_channels'), '<i4')]

    def new_infer_dtype(self):
        return self.dtype



class EventPatternFitSV(straxen.EventPatternFit):
    depends_on = ['event_area_per_channel_sv', 'event_basics_sv', 'event_positions_sv']
    provides = ['event_pattern_fit_sv']
    dtype = [(('Modified Poisson likelihood value for main S2 in the event', 's2_2llh'), '<f4'), (('Data-driven based likelihood value for main S2 in the event', 's2_neural_2llh'), '<f4'), (('Modified Poisson likelihood value for alternative S2', 'alt_s2_2llh'), '<f4'), (('Data-driven based likelihood value for alternative S2 in the event', 'alt_s2_neural_2llh'), '<f4'), (('Modified Poisson likelihood value for main S1', 's1_2llh'), '<f4'), (('Modified Poisson likelihood value for main S1, calculated from top array', 's1_top_2llh'), '<f4'), (('Modified Poisson likelihood value for main S1, calculated from bottom array', 's1_bottom_2llh'), '<f4'), (('Continuous binomial test for S1 area fraction top', 's1_area_fraction_top_continuous_probability'), '<f4'), (('Discrete binomial test for S1 area fraction top', 's1_area_fraction_top_discrete_probability'), '<f4'), (('Continuous binomial test for S1 photon fraction top', 's1_photon_fraction_top_continuous_probability'), '<f4'), (('Discrete binomial test for S1 photon fraction top', 's1_photon_fraction_top_discrete_probability'), '<f4'), (('Continuous binomial test for alternative S1 area fraction top', 'alt_s1_area_fraction_top_continuous_probability'), '<f4'), (('Discrete binomial test for alternative S1 area fraction top', 'alt_s1_area_fraction_top_discrete_probability'), '<f4'), (('Continuous binomial test for alternative S1 photon fraction top', 'alt_s1_photon_fraction_top_continuous_probability'), '<f4'), (('Discrete binomial test for alternative S1 photon fraction top', 'alt_s1_photon_fraction_top_discrete_probability'), '<f4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class EventPositionsSV(straxen.EventPositions):
    depends_on = ['event_basics_sv']
    provides = ['event_positions_sv']
    dtype = [(('Main interaction x-position, field-distortion corrected (cm)', 'x'), '<f4'), (('Alternative S1 interaction (rel. main S2) x-position, field-distortion corrected (cm)', 'alt_s1_x_fdc'), '<f4'), (('Alternative S2 interaction (rel. main S1) x-position, field-distortion corrected (cm)', 'alt_s2_x_fdc'), '<f4'), (('Main interaction y-position, field-distortion corrected (cm)', 'y'), '<f4'), (('Alternative S1 interaction (rel. main S2) y-position, field-distortion corrected (cm)', 'alt_s1_y_fdc'), '<f4'), (('Alternative S2 interaction (rel. main S1) y-position, field-distortion corrected (cm)', 'alt_s2_y_fdc'), '<f4'), (('Main interaction r-position, field-distortion corrected (cm)', 'r'), '<f4'), (('Alternative S1 interaction (rel. main S2) r-position, field-distortion corrected (cm)', 'alt_s1_r_fdc'), '<f4'), (('Alternative S2 interaction (rel. main S1) r-position, field-distortion corrected (cm)', 'alt_s2_r_fdc'), '<f4'), (('Interaction z-position, using mean drift velocity only (cm)', 'z'), '<f4'), (('Alternative S1 z-position (rel. main S2), using mean drift velocity only (cm)', 'alt_s1_z'), '<f4'), (('Alternative S2 z-position (rel. main S1), using mean drift velocity only (cm)', 'alt_s2_z'), '<f4'), (('Main interaction r-position with observed position (cm)', 'r_naive'), '<f4'), (('Alternative S1 interaction (rel. main S2) r-position with observed position (cm)', 'alt_s1_r_naive'), '<f4'), (('Alternative S2 interaction (rel. main S1) r-position with observed position (cm)', 'alt_s2_r_naive'), '<f4'), (('Main interaction z-position with observed position (cm)', 'z_naive'), '<f4'), (('Alternative S1 interaction (rel. main S2) z-position with observed position (cm)', 'alt_s1_z_naive'), '<f4'), (('Alternative S2 interaction (rel. main S1) z-position with observed position (cm)', 'alt_s2_z_naive'), '<f4'), (('Correction added to r_naive for field distortion (cm)', 'r_field_distortion_correction'), '<f4'), (('Correction added to alt_s1_r_naive for field distortion (cm)', 'alt_s1_r_field_distortion_correction'), '<f4'), (('Correction added to alt_s2_r_naive for field distortion (cm)', 'alt_s2_r_field_distortion_correction'), '<f4'), (('Correction added to z_naive for field distortion (cm)', 'z_field_distortion_correction'), '<f4'), (('Correction added to alt_s1_z_naive for field distortion (cm)', 'alt_s1_z_field_distortion_correction'), '<f4'), (('Correction added to alt_s2_z_naive for field distortion (cm)', 'alt_s2_z_field_distortion_correction'), '<f4'), (('Alternative S1 (rel. main S2) interaction angular position (radians)', 'alt_s1_theta'), '<f4'), (('Alternative S2 (rel. main S1) interaction angular position (radians)', 'alt_s2_theta'), '<f4'), (('Main interaction angular position (radians)', 'theta'), '<f4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class EventS2PositionCNNSV(straxen.EventS2PositionCNN):
    depends_on = ['event_area_per_channel_sv', 'event_basics_sv']
    provides = ['event_s2_position_cnn_sv']
    dtype = [(('Reconstructed cnn S2 X position (cm), uncorrected', 'event_s2_x_cnn'), '<f4'), (('Reconstructed cnn S2 Y position (cm), uncorrected', 'event_s2_y_cnn'), '<f4'), (('Reconstructed cnn alt S2 X position (cm), uncorrected', 'event_alt_s2_x_cnn'), '<f4'), (('Reconstructed cnn alt S2 Y position (cm), uncorrected', 'event_alt_s2_y_cnn'), '<f4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class EventS2PositionGCNSV(straxen.EventS2PositionGCN):
    depends_on = ['event_area_per_channel_sv', 'event_basics_sv']
    provides = ['event_s2_position_gcn_sv']
    dtype = [(('Reconstructed gcn S2 X position (cm), uncorrected', 'event_s2_x_gcn'), '<f4'), (('Reconstructed gcn S2 Y position (cm), uncorrected', 'event_s2_y_gcn'), '<f4'), (('Reconstructed gcn alt S2 X position (cm), uncorrected', 'event_alt_s2_x_gcn'), '<f4'), (('Reconstructed gcn alt S2 Y position (cm), uncorrected', 'event_alt_s2_y_gcn'), '<f4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class EventS2PositionMLPSV(straxen.EventS2PositionMLP):
    depends_on = ['event_area_per_channel_sv', 'event_basics_sv']
    provides = ['event_s2_position_mlp_sv']
    dtype = [(('Reconstructed mlp S2 X position (cm), uncorrected', 'event_s2_x_mlp'), '<f4'), (('Reconstructed mlp S2 Y position (cm), uncorrected', 'event_s2_y_mlp'), '<f4'), (('Reconstructed mlp alt S2 X position (cm), uncorrected', 'event_alt_s2_x_mlp'), '<f4'), (('Reconstructed mlp alt S2 Y position (cm), uncorrected', 'event_alt_s2_y_mlp'), '<f4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class EventShadowSV(straxen.EventShadow):
    depends_on = ['event_basics_sv', 'peak_basics_sv', 'peak_shadow_sv']
    provides = ['event_shadow_sv']
    dtype = [(('largest time shadow casting from previous s1 to main S1 [PE/ns]', 's1_shadow_s1_time_shadow'), '<f4'), (('time difference from the previous s1 casting largest time shadow to main S1 [ns]', 's1_dt_s1_time_shadow'), '<i8'), (('time difference from the nearest previous large s1 to main S1 [ns]', 's1_nearest_dt_s1'), '<i8'), (('largest time shadow casting from previous s2 to main S1 [PE/ns]', 's1_shadow_s2_time_shadow'), '<f4'), (('time difference from the previous s2 casting largest time shadow to main S1 [ns]', 's1_dt_s2_time_shadow'), '<i8'), (('x of previous s2 peak casting largest time shadow on main S1 [cm]', 's1_x_s2_time_shadow'), '<f4'), (('y of previous s2 peak casting largest time shadow on main S1 [cm]', 's1_y_s2_time_shadow'), '<f4'), (('time difference from the nearest previous large s2 to main S1 [ns]', 's1_nearest_dt_s2'), '<i8'), (('largest position shadow casting from previous s2 to main S1 [PE/ns]', 's1_shadow_s2_position_shadow'), '<f4'), (('time difference from the previous s2 casting largest position shadow to main S1 [ns]', 's1_dt_s2_position_shadow'), '<i8'), (('x of previous s2 peak casting largest position shadow on main S1 [cm]', 's1_x_s2_position_shadow'), '<f4'), (('y of previous s2 peak casting largest position shadow on main S1 [cm]', 's1_y_s2_position_shadow'), '<f4'), (('PDF describing correlation between previous s2 and main S1', 's1_pdf_s2_position_shadow'), '<f4'), (('largest time shadow casting from previous s1 to main S2 [PE/ns]', 's2_shadow_s1_time_shadow'), '<f4'), (('time difference from the previous s1 casting largest time shadow to main S2 [ns]', 's2_dt_s1_time_shadow'), '<i8'), (('time difference from the nearest previous large s1 to main S2 [ns]', 's2_nearest_dt_s1'), '<i8'), (('largest time shadow casting from previous s2 to main S2 [PE/ns]', 's2_shadow_s2_time_shadow'), '<f4'), (('time difference from the previous s2 casting largest time shadow to main S2 [ns]', 's2_dt_s2_time_shadow'), '<i8'), (('x of previous s2 peak casting largest time shadow on main S2 [cm]', 's2_x_s2_time_shadow'), '<f4'), (('y of previous s2 peak casting largest time shadow on main S2 [cm]', 's2_y_s2_time_shadow'), '<f4'), (('time difference from the nearest previous large s2 to main S2 [ns]', 's2_nearest_dt_s2'), '<i8'), (('largest position shadow casting from previous s2 to main S2 [PE/ns]', 's2_shadow_s2_position_shadow'), '<f4'), (('time difference from the previous s2 casting largest position shadow to main S2 [ns]', 's2_dt_s2_position_shadow'), '<i8'), (('x of previous s2 peak casting largest position shadow on main S2 [cm]', 's2_x_s2_position_shadow'), '<f4'), (('y of previous s2 peak casting largest position shadow on main S2 [cm]', 's2_y_s2_position_shadow'), '<f4'), (('PDF describing correlation between previous s2 and main S2', 's2_pdf_s2_position_shadow'), '<f4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class EventTopBottomParamsSV(straxen.EventTopBottomParams):
    depends_on = ['event_info_sv', 'event_area_per_channel_sv']
    provides = ['event_top_bottom_params_sv']
    dtype = [(('Central time for main S1 for top PMTs [ ns ]', 's1_center_time_top'), '<i8'), (('Time between 10% and 50% area quantiles for main S1 for top PMTs [ns]', 's1_rise_time_top'), '<f4'), (('Width (in ns) of the central 50% area of the peak for top PMTs of main S1', 's1_range_50p_area_top'), '<f4'), (('Width (in ns) of the central 90% area of the peak for top PMTs of main S1', 's1_range_90p_area_top'), '<f4'), (('Central time for main S1 for bot PMTs [ ns ]', 's1_center_time_bot'), '<i8'), (('Time between 10% and 50% area quantiles for main S1 for bot PMTs [ns]', 's1_rise_time_bot'), '<f4'), (('Width (in ns) of the central 50% area of the peak for bot PMTs of main S1', 's1_range_50p_area_bot'), '<f4'), (('Width (in ns) of the central 90% area of the peak for bot PMTs of main S1', 's1_range_90p_area_bot'), '<f4'), (('Difference between center times of top and bottom arrays for main S1 [ ns ]', 's1_center_time_diff_top_bot'), '<i8'), (('Central time for main S2 for top PMTs [ ns ]', 's2_center_time_top'), '<i8'), (('Time between 10% and 50% area quantiles for main S2 for top PMTs [ns]', 's2_rise_time_top'), '<f4'), (('Width (in ns) of the central 50% area of the peak for top PMTs of main S2', 's2_range_50p_area_top'), '<f4'), (('Width (in ns) of the central 90% area of the peak for top PMTs of main S2', 's2_range_90p_area_top'), '<f4'), (('Central time for main S2 for bot PMTs [ ns ]', 's2_center_time_bot'), '<i8'), (('Time between 10% and 50% area quantiles for main S2 for bot PMTs [ns]', 's2_rise_time_bot'), '<f4'), (('Width (in ns) of the central 50% area of the peak for bot PMTs of main S2', 's2_range_50p_area_bot'), '<f4'), (('Width (in ns) of the central 90% area of the peak for bot PMTs of main S2', 's2_range_90p_area_bot'), '<f4'), (('Difference between center times of top and bottom arrays for main S2 [ ns ]', 's2_center_time_diff_top_bot'), '<i8'), (('Central time for alternative S1 for top PMTs [ ns ]', 'alt_s1_center_time_top'), '<i8'), (('Time between 10% and 50% area quantiles for alternative S1 for top PMTs [ns]', 'alt_s1_rise_time_top'), '<f4'), (('Width (in ns) of the central 50% area of the peak for top PMTs of alternative S1', 'alt_s1_range_50p_area_top'), '<f4'), (('Width (in ns) of the central 90% area of the peak for top PMTs of alternative S1', 'alt_s1_range_90p_area_top'), '<f4'), (('Central time for alternative S1 for bot PMTs [ ns ]', 'alt_s1_center_time_bot'), '<i8'), (('Time between 10% and 50% area quantiles for alternative S1 for bot PMTs [ns]', 'alt_s1_rise_time_bot'), '<f4'), (('Width (in ns) of the central 50% area of the peak for bot PMTs of alternative S1', 'alt_s1_range_50p_area_bot'), '<f4'), (('Width (in ns) of the central 90% area of the peak for bot PMTs of alternative S1', 'alt_s1_range_90p_area_bot'), '<f4'), (('Difference between center times of top and bottom arrays for alternative S1 [ ns ]', 'alt_s1_center_time_diff_top_bot'), '<i8'), (('Central time for alternative S2 for top PMTs [ ns ]', 'alt_s2_center_time_top'), '<i8'), (('Time between 10% and 50% area quantiles for alternative S2 for top PMTs [ns]', 'alt_s2_rise_time_top'), '<f4'), (('Width (in ns) of the central 50% area of the peak for top PMTs of alternative S2', 'alt_s2_range_50p_area_top'), '<f4'), (('Width (in ns) of the central 90% area of the peak for top PMTs of alternative S2', 'alt_s2_range_90p_area_top'), '<f4'), (('Central time for alternative S2 for bot PMTs [ ns ]', 'alt_s2_center_time_bot'), '<i8'), (('Time between 10% and 50% area quantiles for alternative S2 for bot PMTs [ns]', 'alt_s2_rise_time_bot'), '<f4'), (('Width (in ns) of the central 50% area of the peak for bot PMTs of alternative S2', 'alt_s2_range_50p_area_bot'), '<f4'), (('Width (in ns) of the central 90% area of the peak for bot PMTs of alternative S2', 'alt_s2_range_90p_area_bot'), '<f4'), (('Difference between center times of top and bottom arrays for alternative S2 [ ns ]', 'alt_s2_center_time_diff_top_bot'), '<i8'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class EventsSV(straxen.Events):
    depends_on = ['peak_basics_sv', 'peak_proximity_sv']
    provides = ['events_sv']
    dtype = [(('Event number in this dataset', 'event_number'), '<i8'), (('Event start time in ns since the unix epoch', 'time'), '<i8'), (('Event end time in ns since the unix epoch', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class EventwBayesClassSV(straxen.EventwBayesClass):
    depends_on = ['peak_classification_bayes_sv', 'event_basics_sv']
    provides = ['event_w_bayes_class_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Given an s1, s1 ln probability', 's1_ln_prob_s1'), '<f4'), (('Given an s1, s2 ln probability', 's1_ln_prob_s2'), '<f4'), (('Given an s2, s1 ln probability', 's2_ln_prob_s1'), '<f4'), (('Given an s2, s2 ln probability', 's2_ln_prob_s2'), '<f4'), (('Given an alt_s1, s1 ln probability', 'alt_s1_ln_prob_s1'), '<f4'), (('Given an alt_s1, s2 ln probability', 'alt_s1_ln_prob_s2'), '<f4'), (('Given an alt_s2, s1 ln probability', 'alt_s2_ln_prob_s1'), '<f4'), (('Given an alt_s2, s2 ln probability', 'alt_s2_ln_prob_s2'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class IndividualPeakMonitorSV(straxen.IndividualPeakMonitor):
    depends_on = ['peak_basics_sv', 'peak_positions_mlp_sv']
    provides = ['individual_peak_monitor_sv']
    dtype = [(('Peak integral in PE', 'area'), '<f4'), (('Reconstructed mlp peak x-position', 'x_mlp'), '<f4'), (('Reconstructed mlp peak y-position', 'y_mlp'), '<f4'), (('Width (in ns) of the central 50% area of the peak', 'range_50p_area'), '<f4'), (('Fraction of original peaks array length that is saved', 'weight'), '<f4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class LEDAfterpulseProcessingSV(straxen.LEDAfterpulseProcessing):
    depends_on = ['raw_records_sv']
    provides = ['afterpulses_sv']
    dtype = [(('Channel/PMT number', 'channel'), '<i2'), (('Time resolution in ns', 'dt'), '<i2'), (('Start time of the interval (ns since unix epoch)', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Integral in ADC x samples', 'area'), '<i4'), (('Pulse area in PE', 'area_pe'), '<f4'), (('Sample index in which hit starts', 'left'), '<i2'), (('Sample index in which hit area succeeds 10% of total area', 'sample_10pc_area'), '<i2'), (('Sample index in which hit area succeeds 50% of total area', 'sample_50pc_area'), '<i2'), (('Sample index of hit maximum', 'max'), '<i2'), (('Index of first sample in record just beyond hit (exclusive bound)', 'right'), '<i2'), (('Height of hit in ADC counts', 'height'), '<i4'), (('Height of hit in PE', 'height_pe'), '<f4'), (('Delay of hit w.r.t. LED hit in same WF, in samples', 'tdelay'), '<i2'), (('Internal (temporary) index of fragment in which hit was found', 'record_i'), '<i4'), (('Index of sample in record where integration starts', 'left_integration'), '<i2'), (('Index of first sample beyond integration region', 'right_integration'), '<i2'), (('ADC threshold applied in order to find hits', 'threshold'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class LEDCalibrationSV(straxen.LEDCalibration):
    depends_on = ['raw_records_sv']
    provides = ['led_calibration_sv']
    dtype = [(('Area averaged in integration windows', 'area'), '<f4'), (('Amplitude in LED window', 'amplitude_led'), '<f4'), (('Amplitude in off LED window', 'amplitude_noise'), '<f4'), (('Channel', 'channel'), '<i2'), (('Start time of the interval (ns since unix epoch)', 'time'), '<i8'), (('Time resolution in ns', 'dt'), '<i2'), (('Length of the interval in samples', 'length'), '<i4')]

    def new_infer_dtype(self):
        return self.dtype



class LocalMinimumInfoSV(straxen.LocalMinimumInfo):
    depends_on = ['event_basics_sv', 'peaks_sv']
    provides = ['event_local_min_info_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Maximum Goodness of Split', 's2_max_gos'), '<f4'), (('Number of local maxima of the smoothed peak', 's2_num_loc_max'), '<i2'), (('Full gap at p% of the valley height of the deepest valley [ns],by default p = 90', 's2_valley_gap'), '<f4'), (('Valley depth over max height of the deepest valley', 's2_valley_height_ratio'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class MergedS2sSV(straxen.MergedS2s):
    depends_on = ['peaklets_sv', 'peaklet_classification_sv', 'lone_hits_sv']
    provides = ['merged_s2s_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i4'), (('Channel/PMT number', 'channel'), '<i2'), (('Classification of the peak(let)', 'type'), 'i1'), (('Waveform data in PE/sample (not PE/ns!), top array', 'data_top'), '<f4', (200,)), (('Integral across channels [PE]', 'area'), '<f4'), (('Integral per channel [PE]', 'area_per_channel'), '<f4', (494,)), (('Number of hits contributing at least one sample to the peak ', 'n_hits'), '<i4'), (('Waveform data in PE/sample (not PE/ns!)', 'data'), '<f4', (200,)), (('Peak widths in range of central area fraction [ns]', 'width'), '<f4', (11,)), (('Peak widths: time between nth and 5th area decile [ns]', 'area_decile_from_midpoint'), '<f4', (11,)), (('Does the channel reach ADC saturation?', 'saturated_channel'), 'i1', (494,)), (('Total number of saturated channels', 'n_saturated_channels'), '<i2'), (('Channel within tight range of mean', 'tight_coincidence'), '<i2'), (('Largest gap between hits inside peak [ns]', 'max_gap'), '<i4'), (('Maximum interior goodness of split', 'max_goodness_of_split'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class MergedS2sHighEnergySV(straxen.MergedS2sHighEnergy):
    depends_on = ['peaklets_he_sv', 'peaklet_classification_he_sv']
    provides = ['merged_s2s_he_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i4'), (('Channel/PMT number', 'channel'), '<i2'), (('Classification of the peak(let)', 'type'), 'i1'), (('Integral across channels [PE]', 'area'), '<f4'), (('Integral per channel [PE]', 'area_per_channel'), '<f4', (752,)), (('Number of hits contributing at least one sample to the peak ', 'n_hits'), '<i4'), (('Waveform data in PE/sample (not PE/ns!)', 'data'), '<f4', (200,)), (('Peak widths in range of central area fraction [ns]', 'width'), '<f4', (11,)), (('Peak widths: time between nth and 5th area decile [ns]', 'area_decile_from_midpoint'), '<f4', (11,)), (('Does the channel reach ADC saturation?', 'saturated_channel'), 'i1', (752,)), (('Total number of saturated channels', 'n_saturated_channels'), '<i2'), (('Channel within tight range of mean', 'tight_coincidence'), '<i2'), (('Largest gap between hits inside peak [ns]', 'max_gap'), '<i4'), (('Maximum interior goodness of split', 'max_goodness_of_split'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class OnlineMonitorMVSV(straxen.OnlineMonitorMV):
    depends_on = ['hitlets_mv_sv', 'events_mv_sv']
    provides = ['online_monitor_mv_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('hitlets_mv per channel', 'hitlets_mv_per_channel'), '<i8', (84,)), (('events_mv_area per chunk', 'events_mv_area_per_chunk'), '<i8', (131,)), (('events_mv per chunk', 'events_mv_per_chunk'), '<i8'), (('events_mv 4-coincidence per chunk', 'events_mv_4coinc_per_chunk'), '<i8'), (('events_mv 5-coincidence per chunk', 'events_mv_5coinc_per_chunk'), '<i8'), (('events_mv 8-coincidence per chunk', 'events_mv_8coinc_per_chunk'), '<i8'), (('events_mv 10-coincidence per chunk', 'events_mv_10coinc_per_chunk'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class OnlineMonitorNVSV(straxen.OnlineMonitorNV):
    depends_on = ['hitlets_nv_sv', 'events_nv_sv']
    provides = ['online_monitor_nv_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('hitlets_nv per channel', 'hitlets_nv_per_channel'), '<i8', (120,)), (('events_nv_area per chunk', 'events_nv_area_per_chunk'), '<i8', (131,)), (('events_nv per chunk', 'events_nv_per_chunk'), '<i8'), (('events_nv 4-coincidence per chunk', 'events_nv_4coinc_per_chunk'), '<i8'), (('events_nv 5-coincidence per chunk', 'events_nv_5coinc_per_chunk'), '<i8'), (('events_nv 8-coincidence per chunk', 'events_nv_8coinc_per_chunk'), '<i8'), (('events_nv 10-coincidence per chunk', 'events_nv_10coinc_per_chunk'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class OnlinePeakMonitorSV(straxen.OnlinePeakMonitor):
    depends_on = ['peak_basics_sv', 'lone_hits_sv']
    provides = ['online_peak_monitor_sv']
    dtype = [(('Start time of the chunk', 'time'), '<i8'), (('End time of the chunk', 'endtime'), '<i8'), (('Area vs width histogram (log-log)', 'area_vs_width_hist'), '<i8', (60, 60)), (('Area vs width edges (log-space)', 'area_vs_width_bounds'), '<f8', (2, 2)), (('Lone hits areas histogram [ADC-counts]', 'lone_hits_area_hist'), '<i8', (100,)), (('Lone hits areas bounds [ADC-counts]', 'lone_hits_area_bounds'), '<f8', (2,)), (('Lone hits per channel', 'lone_hits_per_channel'), '<i8', (494,)), (('AFT histogram', 'aft_hist'), '<i8', (100,)), (('AFT bounds', 'aft_bounds'), '<f8', (2,)), (('Number of contributing channels histogram', 'n_channel_hist'), '<i8', (494,)), (('Single electron gain', 'online_se_gain'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class PeakAmbienceSV(straxen.PeakAmbience):
    depends_on = ['lone_hits_sv', 'peak_basics_sv', 'peak_positions_sv']
    provides = ['peak_ambience_sv']
    dtype = [(('Number of small lh before a peak', 'n_lh_before'), '<i2'), (('Area sum of small lh before a peak', 's_lh_before'), '<f4'), (('Number of small s0 before a peak', 'n_s0_before'), '<i2'), (('Area sum of small s0 before a peak', 's_s0_before'), '<f4'), (('Number of small s1 before a peak', 'n_s1_before'), '<i2'), (('Area sum of small s1 before a peak', 's_s1_before'), '<f4'), (('Number of small s2 before a peak', 'n_s2_before'), '<i2'), (('Area sum of small s2 before a peak', 's_s2_before'), '<f4'), (('Number of small s2 near a peak', 'n_s2_near'), '<i2'), (('Area sum of small s2 near a peak', 's_s2_near'), '<f4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class PeakBasicsSV(straxen.PeakBasics):
    depends_on = ['peaks_sv']
    provides = ['peak_basics_sv']
    dtype = [(('Start time of the peak (ns since unix epoch)', 'time'), '<i8'), (('End time of the peak (ns since unix epoch)', 'endtime'), '<i8'), (('Weighted center time of the peak (ns since unix epoch)', 'center_time'), '<i8'), (('Peak integral in PE', 'area'), '<f4'), (('Number of hits contributing at least one sample to the peak', 'n_hits'), '<i4'), (('Number of PMTs contributing to the peak', 'n_channels'), '<i2'), (('PMT number which contributes the most PE', 'max_pmt'), '<i2'), (('Area of signal in the largest-contributing PMT (PE)', 'max_pmt_area'), '<f4'), (('Total number of saturated channels', 'n_saturated_channels'), '<i2'), (('Width (in ns) of the central 50% area of the peak', 'range_50p_area'), '<f4'), (('Width (in ns) of the central 90% area of the peak', 'range_90p_area'), '<f4'), (('Fraction of area seen by the top array (NaN for peaks with non-positive area)', 'area_fraction_top'), '<f4'), (('Length of the peak waveform in samples', 'length'), '<i4'), (('Time resolution of the peak waveform in ns', 'dt'), '<i2'), (('Time between 10% and 50% area quantiles [ns]', 'rise_time'), '<f4'), (('Number of PMTs with hits within tight range of mean', 'tight_coincidence'), '<i2'), (('Classification of the peak(let)', 'type'), 'i1')]

    def new_infer_dtype(self):
        return self.dtype



class PeakBasicsHighEnergySV(straxen.PeakBasicsHighEnergy):
    depends_on = ['peaks_he_sv']
    provides = ['peak_basics_he_sv']
    dtype = [(('Start time of the peak (ns since unix epoch)', 'time'), '<i8'), (('End time of the peak (ns since unix epoch)', 'endtime'), '<i8'), (('Weighted center time of the peak (ns since unix epoch)', 'center_time'), '<i8'), (('Peak integral in PE', 'area'), '<f4'), (('Number of hits contributing at least one sample to the peak', 'n_hits'), '<i4'), (('Number of PMTs contributing to the peak', 'n_channels'), '<i2'), (('PMT number which contributes the most PE', 'max_pmt'), '<i2'), (('Area of signal in the largest-contributing PMT (PE)', 'max_pmt_area'), '<f4'), (('Total number of saturated channels', 'n_saturated_channels'), '<i2'), (('Width (in ns) of the central 50% area of the peak', 'range_50p_area'), '<f4'), (('Width (in ns) of the central 90% area of the peak', 'range_90p_area'), '<f4'), (('Fraction of area seen by the top array (NaN for peaks with non-positive area)', 'area_fraction_top'), '<f4'), (('Length of the peak waveform in samples', 'length'), '<i4'), (('Time resolution of the peak waveform in ns', 'dt'), '<i2'), (('Time between 10% and 50% area quantiles [ns]', 'rise_time'), '<f4'), (('Number of PMTs with hits within tight range of mean', 'tight_coincidence'), '<i2'), (('Classification of the peak(let)', 'type'), 'i1')]

    def new_infer_dtype(self):
        return self.dtype



class PeakPositionsCNNSV(straxen.PeakPositionsCNN):
    depends_on = ['peaks_sv']
    provides = ['peak_positions_cnn_sv']
    dtype = [(('Reconstructed cnn S2 X position (cm), uncorrected', 'x_cnn'), '<f4'), (('Reconstructed cnn S2 Y position (cm), uncorrected', 'y_cnn'), '<f4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class PeakPositionsGCNSV(straxen.PeakPositionsGCN):
    depends_on = ['peaks_sv']
    provides = ['peak_positions_gcn_sv']
    dtype = [(('Reconstructed gcn S2 X position (cm), uncorrected', 'x_gcn'), '<f4'), (('Reconstructed gcn S2 Y position (cm), uncorrected', 'y_gcn'), '<f4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class PeakPositionsMLPSV(straxen.PeakPositionsMLP):
    depends_on = ['peaks_sv']
    provides = ['peak_positions_mlp_sv']
    dtype = [(('Reconstructed mlp S2 X position (cm), uncorrected', 'x_mlp'), '<f4'), (('Reconstructed mlp S2 Y position (cm), uncorrected', 'y_mlp'), '<f4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class PeakPositionsNTSV(straxen.PeakPositionsNT):
    depends_on = ['peak_positions_cnn_sv', 'peak_positions_mlp_sv', 'peak_positions_gcn_sv']
    provides = ['peak_positions_sv']
    dtype = [(('Reconstructed cnn S2 X position (cm), uncorrected', 'x_cnn'), '<f4'), (('Reconstructed cnn S2 Y position (cm), uncorrected', 'y_cnn'), '<f4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Reconstructed mlp S2 X position (cm), uncorrected', 'x_mlp'), '<f4'), (('Reconstructed mlp S2 Y position (cm), uncorrected', 'y_mlp'), '<f4'), (('Reconstructed gcn S2 X position (cm), uncorrected', 'x_gcn'), '<f4'), (('Reconstructed gcn S2 Y position (cm), uncorrected', 'y_gcn'), '<f4'), (('Reconstructed S2 X position (cm), uncorrected', 'x'), '<f4'), (('Reconstructed S2 Y position (cm), uncorrected', 'y'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class PeakProximitySV(straxen.PeakProximity):
    depends_on = ['peak_basics_sv']
    provides = ['peak_proximity_sv']
    dtype = [(('Number of nearby larger or slightly smaller peaks', 'n_competing'), '<i4'), (('Number of larger or slightly smaller peaks left of the main peak', 'n_competing_left'), '<i4'), (('Time between end of previous peak and start of this peak [ns]', 't_to_prev_peak'), '<i8'), (('Time between end of this peak and start of next peak [ns]', 't_to_next_peak'), '<i8'), (('Smaller of t_to_prev_peak and t_to_next_peak [ns]', 't_to_nearest_peak'), '<i8'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class PeakShadowSV(straxen.PeakShadow):
    depends_on = ['peak_basics_sv', 'peak_positions_sv']
    provides = ['peak_shadow_sv']
    dtype = [(('previous large s1 casted largest time shadow [PE/ns]', 'shadow_s1_time_shadow'), '<f4'), (('time difference to the previous large s1 peak casting largest time shadow [ns]', 'dt_s1_time_shadow'), '<i8'), (('time difference to the nearest previous large s1 [ns]', 'nearest_dt_s1'), '<i8'), (('previous large s2 casted largest time shadow [PE/ns]', 'shadow_s2_time_shadow'), '<f4'), (('time difference to the previous large s2 peak casting largest time shadow [ns]', 'dt_s2_time_shadow'), '<i8'), (('x of previous large s2 peak casting largest time shadow [cm]', 'x_s2_time_shadow'), '<f4'), (('y of previous large s2 peak casting largest time shadow [cm]', 'y_s2_time_shadow'), '<f4'), (('time difference to the nearest previous large s2 [ns]', 'nearest_dt_s2'), '<i8'), (('previous large s2 casted largest position shadow [PE/ns]', 'shadow_s2_position_shadow'), '<f4'), (('time difference to the previous large s2 peak casting largest position shadow [ns]', 'dt_s2_position_shadow'), '<i8'), (('x of previous large s2 peak casting largest position shadow [cm]', 'x_s2_position_shadow'), '<f4'), (('y of previous large s2 peak casting largest position shadow [cm]', 'y_s2_position_shadow'), '<f4'), (('PDF describing correlation to the previous large s2', 'pdf_s2_position_shadow'), '<f4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class PeakTopBottomParamsSV(straxen.PeakTopBottomParams):
    depends_on = ['peaks_sv', 'peak_basics_sv']
    provides = ['peak_top_bottom_params_sv']
    dtype = [(('Central time for top PMTs [ ns ]', 'center_time_top'), '<i8'), (('Time between 10% and 50% area quantiles for top PMTs [ns]', 'rise_time_top'), '<f4'), (('Width (in ns) of the central 50% area of the peak for top PMTs', 'range_50p_area_top'), '<f4'), (('Width (in ns) of the central 90% area of the peak for top PMTs', 'range_90p_area_top'), '<f4'), (('Central time for bot PMTs [ ns ]', 'center_time_bot'), '<i8'), (('Time between 10% and 50% area quantiles for bot PMTs [ns]', 'rise_time_bot'), '<f4'), (('Width (in ns) of the central 50% area of the peak for bot PMTs', 'range_50p_area_bot'), '<f4'), (('Width (in ns) of the central 90% area of the peak for bot PMTs', 'range_90p_area_bot'), '<f4'), (('Difference between center times of top and bottom arrays [ ns ]', 'center_time_diff_top_bot'), '<i8'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class PeakletClassificationSV(straxen.PeakletClassification):
    depends_on = ['peaklets_sv']
    provides = ['peaklet_classification_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i4'), (('Channel/PMT number', 'channel'), '<i2'), (('Classification of the peak(let)', 'type'), 'i1')]

    def new_infer_dtype(self):
        return self.dtype



class PeakletClassificationHighEnergySV(straxen.PeakletClassificationHighEnergy):
    depends_on = ['peaklets_he_sv']
    provides = ['peaklet_classification_he_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i4'), (('Channel/PMT number', 'channel'), '<i2'), (('Classification of the peak(let)', 'type'), 'i1')]

    def new_infer_dtype(self):
        return self.dtype



class PeakletsSV(straxen.Peaklets):
    depends_on = ['records_sv']
    provides = ['peaklets_sv', 'lone_hits_sv']
    dtype = {'peaklets_sv': [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i4'), (('Channel/PMT number', 'channel'), '<i2'), (('Classification of the peak(let)', 'type'), 'i1'), (('Waveform data in PE/sample (not PE/ns!), top array', 'data_top'), '<f4', (200,)), (('Integral across channels [PE]', 'area'), '<f4'), (('Integral per channel [PE]', 'area_per_channel'), '<f4', (494,)), (('Number of hits contributing at least one sample to the peak ', 'n_hits'), '<i4'), (('Waveform data in PE/sample (not PE/ns!)', 'data'), '<f4', (200,)), (('Peak widths in range of central area fraction [ns]', 'width'), '<f4', (11,)), (('Peak widths: time between nth and 5th area decile [ns]', 'area_decile_from_midpoint'), '<f4', (11,)), (('Does the channel reach ADC saturation?', 'saturated_channel'), 'i1', (494,)), (('Total number of saturated channels', 'n_saturated_channels'), '<i2'), (('Channel within tight range of mean', 'tight_coincidence'), '<i2'), (('Largest gap between hits inside peak [ns]', 'max_gap'), '<i4'), (('Maximum interior goodness of split', 'max_goodness_of_split'), '<f4')], 'lone_hits_sv': [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i2'), (('Channel/PMT number', 'channel'), '<i2'), (('Integral [ADC x samples]', 'area'), '<f4'), (('Index of sample in record in which hit starts', 'left'), '<i2'), (('Index of first sample in record just beyond hit (exclusive bound)', 'right'), '<i2'), (('For lone hits, index of sample in record where integration starts', 'left_integration'), '<i2'), (('For lone hits, index of first sample beyond integration region', 'right_integration'), '<i2'), (('Internal (temporary) index of fragment in which hit was found', 'record_i'), '<i4'), (('ADC threshold applied in order to find hits', 'threshold'), '<f4'), (('Maximum amplitude above baseline [ADC counts]', 'height'), '<f4')]}

    def new_infer_dtype(self):
        return self.dtype



class PeakletsHighEnergySV(straxen.PeakletsHighEnergy):
    depends_on = ['records_he_sv']
    provides = ['peaklets_he_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i4'), (('Channel/PMT number', 'channel'), '<i2'), (('Classification of the peak(let)', 'type'), 'i1'), (('Integral across channels [PE]', 'area'), '<f4'), (('Integral per channel [PE]', 'area_per_channel'), '<f4', (752,)), (('Number of hits contributing at least one sample to the peak ', 'n_hits'), '<i4'), (('Waveform data in PE/sample (not PE/ns!)', 'data'), '<f4', (200,)), (('Peak widths in range of central area fraction [ns]', 'width'), '<f4', (11,)), (('Peak widths: time between nth and 5th area decile [ns]', 'area_decile_from_midpoint'), '<f4', (11,)), (('Does the channel reach ADC saturation?', 'saturated_channel'), 'i1', (752,)), (('Total number of saturated channels', 'n_saturated_channels'), '<i2'), (('Channel within tight range of mean', 'tight_coincidence'), '<i2'), (('Largest gap between hits inside peak [ns]', 'max_gap'), '<i4'), (('Maximum interior goodness of split', 'max_goodness_of_split'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class PeaksSV(straxen.Peaks):
    depends_on = ['peaklets_sv', 'peaklet_classification_sv', 'merged_s2s_sv']
    provides = ['peaks_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i4'), (('Channel/PMT number', 'channel'), '<i2'), (('Classification of the peak(let)', 'type'), 'i1'), (('Waveform data in PE/sample (not PE/ns!), top array', 'data_top'), '<f4', (200,)), (('Integral across channels [PE]', 'area'), '<f4'), (('Integral per channel [PE]', 'area_per_channel'), '<f4', (494,)), (('Number of hits contributing at least one sample to the peak ', 'n_hits'), '<i4'), (('Waveform data in PE/sample (not PE/ns!)', 'data'), '<f4', (200,)), (('Peak widths in range of central area fraction [ns]', 'width'), '<f4', (11,)), (('Peak widths: time between nth and 5th area decile [ns]', 'area_decile_from_midpoint'), '<f4', (11,)), (('Does the channel reach ADC saturation?', 'saturated_channel'), 'i1', (494,)), (('Total number of saturated channels', 'n_saturated_channels'), '<i2'), (('Channel within tight range of mean', 'tight_coincidence'), '<i2'), (('Largest gap between hits inside peak [ns]', 'max_gap'), '<i4'), (('Maximum interior goodness of split', 'max_goodness_of_split'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class PeaksHighEnergySV(straxen.PeaksHighEnergy):
    depends_on = ['peaklets_he_sv', 'peaklet_classification_he_sv', 'merged_s2s_he_sv']
    provides = ['peaks_he_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i4'), (('Channel/PMT number', 'channel'), '<i2'), (('Classification of the peak(let)', 'type'), 'i1'), (('Integral across channels [PE]', 'area'), '<f4'), (('Integral per channel [PE]', 'area_per_channel'), '<f4', (752,)), (('Number of hits contributing at least one sample to the peak ', 'n_hits'), '<i4'), (('Waveform data in PE/sample (not PE/ns!)', 'data'), '<f4', (200,)), (('Peak widths in range of central area fraction [ns]', 'width'), '<f4', (11,)), (('Peak widths: time between nth and 5th area decile [ns]', 'area_decile_from_midpoint'), '<f4', (11,)), (('Does the channel reach ADC saturation?', 'saturated_channel'), 'i1', (752,)), (('Total number of saturated channels', 'n_saturated_channels'), '<i2'), (('Channel within tight range of mean', 'tight_coincidence'), '<i2'), (('Largest gap between hits inside peak [ns]', 'max_gap'), '<i4'), (('Maximum interior goodness of split', 'max_goodness_of_split'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class PulseProcessingSV(straxen.PulseProcessing):
    depends_on = ['raw_records_sv']
    provides = ['records_sv', 'veto_regions_sv', 'pulse_counts_sv']
    dtype = {'records_sv': [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i2'), (('Channel/PMT number', 'channel'), '<i2'), (('Length of pulse to which the record belongs (without zero-padding)', 'pulse_length'), '<i4'), (('Fragment number in the pulse', 'record_i'), '<i2'), (('Integral in ADC counts x samples', 'area'), '<i4'), (('Level of data reduction applied (strax.ReductionLevel enum)', 'reduction_level'), 'u1'), (('Baseline in ADC counts. data = int(baseline) - data_orig', 'baseline'), '<f4'), (('Baseline RMS in ADC counts. data = baseline - data_orig', 'baseline_rms'), '<f4'), (('Multiply data by 2**(this number). Baseline is unaffected.', 'amplitude_bit_shift'), '<i2'), (('Waveform data in raw counts above integer part of baseline', 'data'), '<i2', (110,))], 'veto_regions_sv': [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i2'), (('Channel/PMT number', 'channel'), '<i2'), (('Integral [ADC x samples]', 'area'), '<f4'), (('Index of sample in record in which hit starts', 'left'), '<i2'), (('Index of first sample in record just beyond hit (exclusive bound)', 'right'), '<i2'), (('For lone hits, index of sample in record where integration starts', 'left_integration'), '<i2'), (('For lone hits, index of first sample beyond integration region', 'right_integration'), '<i2'), (('Internal (temporary) index of fragment in which hit was found', 'record_i'), '<i4'), (('ADC threshold applied in order to find hits', 'threshold'), '<f4'), (('Maximum amplitude above baseline [ADC counts]', 'height'), '<f4')], 'pulse_counts_sv': [(('Start time of the chunk', 'time'), '<i8'), (('End time of the chunk', 'endtime'), '<i8'), (('Number of pulses', 'pulse_count'), '<i8', (494,)), (('Number of lone pulses', 'lone_pulse_count'), '<i8', (494,)), (('Integral of all pulses in ADC_count x samples', 'pulse_area'), '<i8', (494,)), (('Integral of lone pulses in ADC_count x samples', 'lone_pulse_area'), '<i8', (494,)), (('Average baseline', 'baseline_mean'), '<i2', (494,)), (('Average baseline rms', 'baseline_rms_mean'), '<f4', (494,))]}

    def new_infer_dtype(self):
        return self.dtype


    save_when = immutabledict(records_sv=strax.SaveWhen.TARGET, veto_regions_sv=strax.SaveWhen.TARGET, pulse_counts_sv=strax.SaveWhen.ALWAYS)

    rechunk_on_save = immutabledict(records_sv= False, veto_regions_sv= True, pulse_counts_sv= True)


class PulseProcessingHighEnergySV(straxen.PulseProcessingHighEnergy):
    depends_on = ['raw_records_he_sv']
    provides = ['records_he_sv', 'pulse_counts_he_sv']
    dtype = {'records_he_sv': [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i2'), (('Channel/PMT number', 'channel'), '<i2'), (('Length of pulse to which the record belongs (without zero-padding)', 'pulse_length'), '<i4'), (('Fragment number in the pulse', 'record_i'), '<i2'), (('Integral in ADC counts x samples', 'area'), '<i4'), (('Level of data reduction applied (strax.ReductionLevel enum)', 'reduction_level'), 'u1'), (('Baseline in ADC counts. data = int(baseline) - data_orig', 'baseline'), '<f4'), (('Baseline RMS in ADC counts. data = baseline - data_orig', 'baseline_rms'), '<f4'), (('Multiply data by 2**(this number). Baseline is unaffected.', 'amplitude_bit_shift'), '<i2'), (('Waveform data in raw counts above integer part of baseline', 'data'), '<i2', (110,))], 'pulse_counts_he_sv': [(('Start time of the chunk', 'time'), '<i8'), (('End time of the chunk', 'endtime'), '<i8'), (('Number of pulses', 'pulse_count'), '<i8', (752,)), (('Number of lone pulses', 'lone_pulse_count'), '<i8', (752,)), (('Integral of all pulses in ADC_count x samples', 'pulse_area'), '<i8', (752,)), (('Integral of lone pulses in ADC_count x samples', 'lone_pulse_area'), '<i8', (752,)), (('Average baseline', 'baseline_mean'), '<i2', (752,)), (('Average baseline rms', 'baseline_rms_mean'), '<f4', (752,))]}

    def new_infer_dtype(self):
        return self.dtype


    rechunk_on_save = immutabledict(records_he_sv= False, pulse_counts_he_sv= True)


class S2ReconPosDiffSV(straxen.S2ReconPosDiff):
    depends_on = ['event_basics_sv']
    provides = ['s2_recon_pos_diff_sv']
    dtype = [(('Mean value of x for main S2', 's2_recon_avg_x'), '<f4'), (('Mean value of x for alternatice S2', 'alt_s2_recon_avg_x'), '<f4'), (('Mean value of y for main S2', 's2_recon_avg_y'), '<f4'), (('Mean value of y for alternatice S2', 'alt_s2_recon_avg_y'), '<f4'), (('Reconstructed position difference for main S2', 's2_recon_pos_diff'), '<f4'), (('Reconstructed position difference for alternative S2', 'alt_s2_recon_pos_diff'), '<f4'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class VetoIntervalsSV(straxen.VetoIntervals):
    depends_on = ['aqmon_hits_sv']
    provides = ['veto_intervals_sv']
    dtype = [(('veto interval [ns]', 'veto_interval'), '<i8'), (('veto signal type', 'veto_type'), '<U30'), (('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class VetoProximitySV(straxen.VetoProximity):
    depends_on = ['event_basics_sv', 'veto_intervals_sv']
    provides = ['veto_proximity_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Duration of event overlapping with "busy"-veto [ns]', 'veto_busy_overlap'), '<i8'), (('Time (absolute value) to previous "busy"-veto from "time" of event [ns]', 'time_to_previous_busy'), '<i8'), (('Time (absolute value) to next "busy"-veto from "endtime" of event [ns]', 'time_to_next_busy'), '<i8'), (('Duration of event overlapping with "busy_he"-veto [ns]', 'veto_busy_he_overlap'), '<i8'), (('Time (absolute value) to previous "busy_he"-veto from "time" of event [ns]', 'time_to_previous_busy_he'), '<i8'), (('Time (absolute value) to next "busy_he"-veto from "endtime" of event [ns]', 'time_to_next_busy_he'), '<i8'), (('Duration of event overlapping with "hev"-veto [ns]', 'veto_hev_overlap'), '<i8'), (('Time (absolute value) to previous "hev"-veto from "time" of event [ns]', 'time_to_previous_hev'), '<i8'), (('Time (absolute value) to next "hev"-veto from "endtime" of event [ns]', 'time_to_next_hev'), '<i8'), (('Duration of event overlapping with "straxen_deadtime"-veto [ns]', 'veto_straxen_deadtime_overlap'), '<i8'), (('Time (absolute value) to previous "straxen_deadtime"-veto from "time" of event [ns]', 'time_to_previous_straxen_deadtime'), '<i8'), (('Time (absolute value) to next "straxen_deadtime"-veto from "endtime" of event [ns]', 'time_to_next_straxen_deadtime'), '<i8'), (('Duration of event overlapping with "software"-veto [ns]', 'veto_software_overlap'), '<i8'), (('Time (absolute value) to previous "software"-veto from "time" of event [ns]', 'time_to_previous_software'), '<i8'), (('Time (absolute value) to next "software"-veto from "endtime" of event [ns]', 'time_to_next_software'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class mVETOEventSyncSV(straxen.mVETOEventSync):
    depends_on = ['events_mv_sv', 'detector_time_offsets_sv']
    provides = ['events_sync_mv_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Time of the event synchronized according to the total digitizer delay.', 'time_sync'), '<i8'), (('Endtime of the event synchronized according to the total digitizer delay.', 'endtime_sync'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class muVETOEventsSV(straxen.muVETOEvents):
    depends_on = ['hitlets_mv_sv']
    provides = ['events_mv_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Veto event number in this dataset', 'event_number_mv'), '<i8'), (('Total area of all hitlets in event [pe]', 'area'), '<f4'), (('Total number of hitlets in events', 'n_hits'), '<i4'), (('Total number of contributing channels', 'n_contributing_pmt'), 'u1'), (('Area in event per channel [pe]', 'area_per_channel'), '<f4', (84,)), (('Area weighted mean time of the event relative to the event start [ns]', 'center_time'), '<f4'), (('Weighted variance of time [ns]', 'center_time_spread'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class muVETOHitletsSV(straxen.muVETOHitlets):
    depends_on = ['records_mv_sv']
    provides = ['hitlets_mv_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i2'), (('Channel/PMT number', 'channel'), '<i2'), (('Total hit area in pe', 'area'), '<f4'), (('Maximum of the PMT pulse in pe/sample', 'amplitude'), '<f4'), (('Position of the Amplitude in ns (minus "time")', 'time_amplitude'), '<i2'), (('Hit entropy', 'entropy'), '<f4'), (('Width (in ns) of the central 50% area of the hitlet', 'range_50p_area'), '<f4'), (('Width (in ns) of the central 80% area of the hitlet', 'range_80p_area'), '<f4'), (('Position of the 25% area decile [ns]', 'left_area'), '<f4'), (('Position of the 10% area decile [ns]', 'low_left_area'), '<f4'), (('Width (in ns) of the highest density region covering a 50% area of the hitlet', 'range_hdr_50p_area'), '<f4'), (('Width (in ns) of the highest density region covering a 80% area of the hitlet', 'range_hdr_80p_area'), '<f4'), (('Left edge of the 50% highest density region  [ns]', 'left_hdr'), '<f4'), (('Left edge of the 80% highest density region  [ns]', 'low_left_hdr'), '<f4'), (('FWHM of the PMT pulse [ns]', 'fwhm'), '<f4'), (('Left edge of the FWHM [ns] (minus "time")', 'left'), '<f4'), (('FWTM of the PMT pulse [ns]', 'fwtm'), '<f4'), (('Left edge of the FWTM [ns] (minus "time")', 'low_left'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class muVETOPulseProcessingSV(straxen.muVETOPulseProcessing):
    depends_on = ['raw_records_mv_sv']
    provides = ['records_mv_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i2'), (('Channel/PMT number', 'channel'), '<i2'), (('Length of pulse to which the record belongs (without zero-padding)', 'pulse_length'), '<i4'), (('Fragment number in the pulse', 'record_i'), '<i2'), (('Integral in ADC counts x samples', 'area'), '<i4'), (('Level of data reduction applied (strax.ReductionLevel enum)', 'reduction_level'), 'u1'), (('Baseline in ADC counts. data = int(baseline) - data_orig', 'baseline'), '<f4'), (('Baseline RMS in ADC counts. data = baseline - data_orig', 'baseline_rms'), '<f4'), (('Multiply data by 2**(this number). Baseline is unaffected.', 'amplitude_bit_shift'), '<i2'), (('Waveform data in raw counts above integer part of baseline', 'data'), '<i2', (110,))]

    def new_infer_dtype(self):
        return self.dtype



class nVETOEventPositionsSV(straxen.nVETOEventPositions):
    depends_on = ['events_nv_sv', 'hitlets_nv_sv']
    provides = ['event_positions_nv_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Number of prompt hitlets within the first "position_max_time_nv" ns of the event.', 'n_prompt_hitlets'), '<i2'), (('Azimuthal angle, where the neutron capture was detected in [0, 2 pi).', 'angle'), '<f4'), (('Area weighted mean of position in x [mm]', 'pos_x'), '<f4'), (('Area weighted mean of position in y [mm]', 'pos_y'), '<f4'), (('Area weighted mean of position in z [mm]', 'pos_z'), '<f4'), (('Weighted variance of position in x [mm]', 'pos_x_spread'), '<f4'), (('Weighted variance of position in y [mm]', 'pos_y_spread'), '<f4'), (('Weighted variance of position in z [mm]', 'pos_z_spread'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class nVETOEventsSV(straxen.nVETOEvents):
    depends_on = ['hitlets_nv_sv']
    provides = ['events_nv_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Veto event number in this dataset', 'event_number_nv'), '<i8'), (('Total area of all hitlets in event [pe]', 'area'), '<f4'), (('Total number of hitlets in events', 'n_hits'), '<i4'), (('Total number of contributing channels', 'n_contributing_pmt'), 'u1'), (('Area in event per channel [pe]', 'area_per_channel'), '<f4', (120,)), (('Area weighted mean time of the event relative to the event start [ns]', 'center_time'), '<f4'), (('Weighted variance of time [ns]', 'center_time_spread'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class nVETOEventsSyncSV(straxen.nVETOEventsSync):
    depends_on = ['events_nv_sv', 'detector_time_offsets_sv']
    provides = ['events_sync_nv_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Exclusive end time since unix epoch [ns]', 'endtime'), '<i8'), (('Time of the event synchronized according to the total digitizer delay.', 'time_sync'), '<i8'), (('Endtime of the event synchronized according to the total digitizer delay.', 'endtime_sync'), '<i8')]

    def new_infer_dtype(self):
        return self.dtype



class nVETOHitletsSV(straxen.nVETOHitlets):
    depends_on = ['records_nv_sv']
    provides = ['hitlets_nv_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i2'), (('Channel/PMT number', 'channel'), '<i2'), (('Total hit area in pe', 'area'), '<f4'), (('Maximum of the PMT pulse in pe/sample', 'amplitude'), '<f4'), (('Position of the Amplitude in ns (minus "time")', 'time_amplitude'), '<i2'), (('Hit entropy', 'entropy'), '<f4'), (('Width (in ns) of the central 50% area of the hitlet', 'range_50p_area'), '<f4'), (('Width (in ns) of the central 80% area of the hitlet', 'range_80p_area'), '<f4'), (('Position of the 25% area decile [ns]', 'left_area'), '<f4'), (('Position of the 10% area decile [ns]', 'low_left_area'), '<f4'), (('Width (in ns) of the highest density region covering a 50% area of the hitlet', 'range_hdr_50p_area'), '<f4'), (('Width (in ns) of the highest density region covering a 80% area of the hitlet', 'range_hdr_80p_area'), '<f4'), (('Left edge of the 50% highest density region  [ns]', 'left_hdr'), '<f4'), (('Left edge of the 80% highest density region  [ns]', 'low_left_hdr'), '<f4'), (('FWHM of the PMT pulse [ns]', 'fwhm'), '<f4'), (('Left edge of the FWHM [ns] (minus "time")', 'left'), '<f4'), (('FWTM of the PMT pulse [ns]', 'fwtm'), '<f4'), (('Left edge of the FWTM [ns] (minus "time")', 'low_left'), '<f4')]

    def new_infer_dtype(self):
        return self.dtype



class nVETOPulseProcessingSV(straxen.nVETOPulseProcessing):
    depends_on = ['raw_records_coin_nv_sv']
    provides = ['records_nv_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i2'), (('Channel/PMT number', 'channel'), '<i2'), (('Length of pulse to which the record belongs (without zero-padding)', 'pulse_length'), '<i4'), (('Fragment number in the pulse', 'record_i'), '<i2'), (('Integral in ADC counts x samples', 'area'), '<i4'), (('Level of data reduction applied (strax.ReductionLevel enum)', 'reduction_level'), 'u1'), (('Baseline in ADC counts. data = int(baseline) - data_orig', 'baseline'), '<f4'), (('Baseline RMS in ADC counts. data = baseline - data_orig', 'baseline_rms'), '<f4'), (('Multiply data by 2**(this number). Baseline is unaffected.', 'amplitude_bit_shift'), '<i2'), (('Waveform data in raw counts above integer part of baseline', 'data'), '<i2', (110,))]

    def new_infer_dtype(self):
        return self.dtype



class nVETORecorderSV(straxen.nVETORecorder):
    depends_on = ['raw_records_nv_sv']
    provides = ['raw_records_coin_nv_sv', 'lone_raw_records_nv_sv', 'lone_raw_record_statistics_nv_sv']
    dtype = {'raw_records_coin_nv_sv': [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i2'), (('Channel/PMT number', 'channel'), '<i2'), (('Length of pulse to which the record belongs (without zero-padding)', 'pulse_length'), '<i4'), (('Fragment number in the pulse', 'record_i'), '<i2'), (('Baseline determined by the digitizer (if this is supported)', 'baseline'), '<i2'), (('Waveform data in raw ADC counts', 'data'), '<i2', (110,))], 'lone_raw_records_nv_sv': [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i2'), (('Channel/PMT number', 'channel'), '<i2'), (('Length of pulse to which the record belongs (without zero-padding)', 'pulse_length'), '<i4'), (('Fragment number in the pulse', 'record_i'), '<i2'), (('Integral in ADC counts x samples', 'area'), '<i4'), (('Level of data reduction applied (strax.ReductionLevel enum)', 'reduction_level'), 'u1'), (('Baseline in ADC counts. data = int(baseline) - data_orig', 'baseline'), '<f4'), (('Baseline RMS in ADC counts. data = baseline - data_orig', 'baseline_rms'), '<f4'), (('Multiply data by 2**(this number). Baseline is unaffected.', 'amplitude_bit_shift'), '<i2'), (('Waveform data in raw counts above integer part of baseline', 'data'), '<i2', (110,))], 'lone_raw_record_statistics_nv_sv': [(('Start time of the chunk', 'time'), '<i8'), (('Endtime of the chunk', 'endtime'), '<i8'), (('Channel of the lone record', 'channel'), '<i4', (120,)), (('Total number of lone record fragments', 'nfragments'), '<i4', (120,)), (('Number of higher order lone fragments', 'nhigherfragments'), '<f8', (120,)), (('Average area per waveform in ADC_count x samples', 'lone_record_area'), '<f8', (120,)), (('Average area of higher fragment lone records in ADC_count x samples', 'higher_lone_record_area'), '<i8', (120,)), (('Baseline mean of lone records in ADC_count', 'baseline_mean'), '<f8', (120,)), (('Baseline spread of lone records in ADC_count', 'baseline_rms'), '<f8', (120,))]}

    def new_infer_dtype(self):
        return self.dtype


    save_when = immutabledict(raw_records_coin_nv_sv=strax.SaveWhen.TARGET, lone_raw_records_nv_sv=strax.SaveWhen.TARGET, lone_raw_record_statistics_nv_sv=strax.SaveWhen.ALWAYS)


class nVetoExtTimingsSV(straxen.nVetoExtTimings):
    depends_on = ['raw_records_nv_sv', 'hitlets_nv_sv']
    provides = ['ext_timings_nv_sv']
    dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'), (('Length of the interval in samples', 'length'), '<i4'), (('Width of one sample [ns]', 'dt'), '<i2'), (('Delta time from trigger timing [ns]', 'delta_time'), '<i2'), (('Index to which pulse (not record) the hitlet belongs to.', 'pulse_i'), '<i4')]

    def new_infer_dtype(self):
        return self.dtype

