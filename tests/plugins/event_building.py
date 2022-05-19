import os
import tempfile
from _core import PluginTestAccumulator, PluginTestCase
import numpy as np
import pandas as pd
from datetime import datetime


@PluginTestAccumulator.register('test_exclude_s1_as_triggering_peaks_config')
def exclude_s1_as_triggering_peaks_config(self: PluginTestCase, trigger_min_area=10):
    """
    Test for checking the event building options, specifically the S1 (exclusion)
    """
    # Change the config to allow s1s to be triggering, also drastically
    # decrease the trigger_min_area to allow small s1s to be triggering
    st = self.st.new_context()
    st.set_config(dict(exclude_s1_as_triggering_peaks=0,
                       trigger_min_area=trigger_min_area,
                       ))
    events = st.get_array(self.run_id, 'event_basics', progress_bar=False)
    new_min_coincidence = int(np.max(events['s1_tight_coincidence']) - 1)

    # Create an alternative config, but with less
    st_alt = st.new_context()
    st_alt.set_config(dict(event_s1_min_coincidence=new_min_coincidence))
    events_alt = st_alt.get_array(self.run_id, 'event_basics', progress_bar=False)

    # The event durations should be smaller, since almost no s1s are
    # considered triggering so the events are extended less in the alt
    # config.
    self.assertLessEqual(
        np.mean(events_alt['endtime'] - events_alt['time']),
        np.mean(events['endtime'] - events['time']),
    )

    # Check the triggers of the alternate config
    event_plugin = st_alt.get_single_plugin(self.run_id, 'events')
    triggers = get_triggering_peaks(events_alt,
                                    event_plugin.left_extension,
                                    event_plugin.right_extension)
    s1_triggers = triggers[triggers['type'] == 1]
    self.assertTrue(all(s1_triggers['tight_coincidence'] >= new_min_coincidence))
    self.assertTrue(all(triggers['area'] >= trigger_min_area))


@PluginTestAccumulator.register('test_partitioned_tpc_corrected_areas')
def test_corrected_areas(self: PluginTestCase, ab_value=20, cd_value=21):
    """
    Run the test in ../test_url_config.TestURLConfig.test_seg_file_json
    on corrected_areas
    """
    fake_file = {'time': [datetime(2000, 1, 1).timestamp() * 1e9,
                          datetime(2021, 1, 1).timestamp() * 1e9,
                          datetime(2040, 1, 1).timestamp() * 1e9],
                 'ab': [10, ab_value, 30],
                 'cd': [11, cd_value, 31]
                 }
    # This example also works well with dataframes!
    temp_dir = tempfile.TemporaryDirectory()
    fake_file_name = os.path.join(temp_dir.name, 'test_seg.csv')
    pd.DataFrame(fake_file).to_csv(fake_file_name)

    self.st.set_config({'se_gain': f'itp_dict://'
                                   f'resource://'
                                   f'{fake_file_name}'
                                   f'?run_id=plugin.run_id'
                                   f'&fmt=csv'
                                   f'&itp_keys=ab,cd'})
    # Try loading some new data with the interpolated dictionary
    _ = self.st.get_array(self.run_id, 'corrected_areas')
    temp_dir.cleanup()


def get_triggering_peaks(events, left_extension, right_extension):
    """
    Extract the first and last triggering peaks from an event and return type, area an tight_coincidence
    """
    peaks = np.zeros(len(events) * 4,
                     dtype=[(('type', 'type of peak'), np.int8),
                            (('tight_coincidence', 'tc level'), np.int16,),
                            (('area', 'peak area'), np.float32),
                            ])
    peaks_seen = 0
    for event in events:
        for peak in 's1_ s2_ alt_s1_ alt_s2_'.split():
            # We know that the event boundaries are set by the first/last
            # triggering peaks, so just select those peaks that have defined the
            # boundary (if we can still). This is not a complete list of
            # triggers, but works well enough for this test
            is_first_trigger = event[f'{peak}time'] - left_extension == event['time']
            is_last_trigger = event[f'{peak}endtime'] + right_extension == event['endtime']
            if is_first_trigger or is_last_trigger:
                peaks[peaks_seen]['type'] = int(peak[-2])
                for field in 'area tight_coincidence'.split():
                    peaks[peaks_seen][field] = event[f'{peak}{field}']
                peaks_seen += 1
    return peaks[:peaks_seen]