from os import path
import pickle

import pytest
from expects import expect, equal
from doubles import allow
import numpy as np
import pandas as pd

from credictor.predictor import CreditsPredictor
from credictor.cvlib.video import Video


def video_path(file_name: str):
    script_dir = path.dirname(__file__)
    rel_path = f'resources/{file_name}'
    return path.join(script_dir, rel_path)


def generate_prediction():
    result = CreditsPredictor().predict(video_path('[path_to_video]'))
    pickle.dump(result, open('[file_name].pkl', 'wb'))


@pytest.fixture
def media_1_features():
    return np.load('credictor/tests/resources/updated_features_8642237.npy')


@pytest.fixture
def media_2_features():
    return np.load('credictor/tests/resources/updated_features_7317500.npy')


class TestCreditsPredictor:
    def test_positive_output(self, media_2_features):
        predictor = CreditsPredictor()
        video = Video('credictor/tests/resources/7317500.mp4', read_every=10, start_at=0.5)
        normalized_features = media_2_features
        candidates = predictor.predict_frames(normalized_features)
        frame_prediction = predictor.find_credits_start_frame(candidates, video.fps(), video.length(), video.cropped_proportion, video.frame_reading_window)
        result = predictor._frame_to_milliseconds(frame_prediction, video.fps(), video.frame_reading_window)
        expect(result > 0).to(equal(True))
    
    def test_time_in_milliseconds_conversion(self, media_2_features):
        predictor = CreditsPredictor()
        video = Video('credictor/tests/resources/7317500.mp4', read_every=10, start_at=0.5)
        normalized_features = media_2_features
        candidates = predictor.predict_frames(normalized_features)
        frame_prediction = predictor.find_credits_start_frame(candidates, video.fps(), video.length(), video.cropped_proportion, video.frame_reading_window)
        result = predictor._frame_to_milliseconds(frame_prediction, video.fps(), video.frame_reading_window)
        expect(result).to(equal(2428094))

    def test_find_credits_start_frame_for_credits_with_background_movement(self, media_1_features):
        predictor = CreditsPredictor()
        video = Video('credictor/tests/resources/8642237.mp4', read_every=10, start_at=0.5)
        preds = predictor.predict_frames(media_1_features)
        result = predictor.find_credits_start_frame(preds, video.fps(), video.length(), video.cropped_proportion, video.frame_reading_window)
        expect(result).to(equal(6890))

    def test_find_credits_start_frame_for_standard_credits(self, media_2_features):
        predictor = CreditsPredictor()
        video = Video('credictor/tests/resources/7317500.mp4', read_every=10, start_at=0.5)
        preds = predictor.predict_frames(media_2_features)
        result = predictor.find_credits_start_frame(preds, video.fps(), video.length(), video.cropped_proportion, video.frame_reading_window)
        expect(result).to(equal(72770))
