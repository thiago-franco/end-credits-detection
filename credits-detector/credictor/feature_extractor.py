import collections
import csv
from typing import List

import pandas as pd

from credictor.cvlib.video import Video
from credictor.cvlib.flow import AverageFlow
from credictor.cvlib.rectangles import Rectangle
from credictor.cvlib.frame_diff import pixel_norm_diff, hist_norm_diff


Cuepoint = collections.namedtuple("Cuepoint", ["start", "end"])


class FeatureExtractor:
    @property
    def features(self) -> List[str]:
        return ['pixel_norm_diff', 'hist_norm_diff', 'nb_flow_points', 'nb_rectangles', 'is_credit']

    def extract(self, video: Video) -> pd.DataFrame:
        last_frame = None
        avg_flow = AverageFlow()

        rows = []
        for i, frame in enumerate(video):
            x_flow, y_flow, nb_flow_points = avg_flow.compute(frame._array)
            rect = Rectangle(frame)
            if last_frame is not None:
                p_diff = pixel_norm_diff(frame, last_frame)
                h_diff = hist_norm_diff(frame, last_frame)
                nb_rectangles = rect.count()
                row_dict = {'pixel_norm_diff': p_diff, 'hist_norm_diff': h_diff, 'nb_flow_points': nb_flow_points, 'nb_rectangles': nb_rectangles}
                rows.append(row_dict)
            last_frame = frame

        features = self.shift_frames(rows)

        return features

    def shift_frames(self, rows):
        shift = 10
        features_df = pd.DataFrame(rows)
        features = features_df
        for i in range(shift):
            shifted = features_df.shift(i + 1)
            shifted.columns += ('_' + str(i + 1))
            features = features.join(shifted)
        return features
