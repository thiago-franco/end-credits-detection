import os
from typing import Optional

from scipy.signal import medfilt
import numpy as np
from pandas import DataFrame, read_pickle

from credictor.feature_extractor import FeatureExtractor
from credictor.cvlib.video import Video

FRAMES_PROCESSING_WINDOW = 10
PROPORTION_TO_CROP = 0.5

class CreditsPredictor:
    def __init__(self):
        self._model = self._load_model('model.pkl')
        self._normalizer = self._load_model('normalizer.pkl')
        self._extractor = FeatureExtractor()

    def _load_model(self, file: str):
        dir, filename = os.path.split(__file__)
        model_path = os.path.join(dir, 'classifier', file)
        return read_pickle(model_path)

    def predict(self, video_path: str) -> int:
        video = Video(video_path, read_every=FRAMES_PROCESSING_WINDOW, start_at=PROPORTION_TO_CROP)
        fps = video.fps()  # fps needs to be obtained before processing video
        length = video.length() # length needs to be obtained before processing video
        normalized_features = self.get_normalized_features(video)
        candidates = self.predict_frames(normalized_features)
        frame_prediction = self.find_credits_start_frame(candidates, fps, length, video.cropped_proportion, video.frame_reading_window)
        prediction = self._frame_to_milliseconds(frame_prediction, fps, video.frame_reading_window)
        return prediction

    def _frame_to_milliseconds(self, frame: int, fps: float, process_every: int) -> int:
        if not (frame and fps and process_every):
            return 0
        return int(frame / fps * 1000)

    def get_normalized_features(self, video: Video) -> np.ndarray:
        features = self._extractor.extract(video)
        x = features.fillna(0)
        normalized_features = self._normalizer.transform(x)
        return normalized_features

    def predict_frames(self, features: np.ndarray) -> np.ndarray:
        accepted_probabilities = self._model.predict_proba(features)[:, 1] > 0.45
        return medfilt(accepted_probabilities, kernel_size=25)

    def find_credits_start_frame(self, preds, fps: float, length: int, start_at: float, process_every: int, crop_pred_prop=0.25, min_credit_running_time = 10) -> Optional[int]:
        crop_pred_end = int(length / process_every * crop_pred_prop)
        cropped_pred = preds[-crop_pred_end:]
        run_values, run_starts, run_lengths = self.find_runs(cropped_pred)
        proposal_starts = run_starts[run_values == 1]
        proposal_lengths = run_lengths[run_values == 1]

        if len(proposal_starts) == 0:
            return None

        return self._winning_start(crop_pred_end, preds, proposal_lengths, proposal_starts, min_credit_running_time, fps, length, start_at, process_every)

    def _winning_start(self, crop_pred_end, preds, proposal_lengths, proposal_starts, min_credit_running_time, fps: float, length: int, start_at: float, process_every: int):
        min_run_length = int((fps / process_every) * min_credit_running_time)
        length_argmax = proposal_lengths.argmax()
        while length_argmax < len(proposal_starts) - 1:
            if (proposal_starts[length_argmax + 1] - proposal_starts[length_argmax]) > (2 * min_run_length):
                length_argmax += 1
            else:
                break
        winning_proposal_start = proposal_starts[length_argmax]
        winning_proposal_length = proposal_lengths[length_argmax]
        if winning_proposal_length >= min_run_length:
            winning_start = winning_proposal_start + self._adjust_winning_start(preds, crop_pred_end, length, start_at, process_every)
            if winning_start >= 0:
                return winning_start * process_every
        else:
            return None
        
    def _adjust_winning_start(self, preds, crop_pred_end: int, length: int, start_at: float, process_every: int):
        return int(length * start_at / process_every) + (len(preds) - crop_pred_end)

    def find_runs(self, predictions: DataFrame) -> tuple:
        n = predictions.shape[0]
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(predictions[:-1], predictions[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]
        run_values = predictions[loc_run_start]
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths
