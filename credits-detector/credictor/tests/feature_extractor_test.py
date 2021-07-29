from os import path
from os.path import basename

from pandas import DataFrame
from expects import expect, be_a, be_true

from credictor.feature_extractor import FeatureExtractor
from credictor.cvlib.video import Video


class TestFeatureExtractor:
    def test_returns_dataframe(self):
        extractor = FeatureExtractor()
        video = Video('credictor/tests/resources/regions3.png')
        features = extractor.extract(video)
        expect(features).to(be_a(DataFrame))

    def test_has_every_feature_columns_when_cuepoints_provided(self):
        extractor = FeatureExtractor()
        video = Video('credictor/tests/resources/regions3.png')
        features_df = extractor.extract(video)
        expect(all(feature in extractor.features for feature in features_df.columns)).to(be_true)
