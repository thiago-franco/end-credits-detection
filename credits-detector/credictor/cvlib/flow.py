import cv2


class AverageFlow(object):
    def __init__(self):
        # params for ShiTomasi corner detection
        self._feature_params = dict(maxCorners=100,
                                    qualityLevel=0.3,
                                    minDistance=7,
                                    blockSize=7)

        # Parameters for lucas kanade optical flow
        self._lk_params = dict(winSize=(15, 15),
                               maxLevel=2,
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self._previous_state = None

    def compute(self, frame):
        if self._previous_state is None or self._previous_state[1] is None or len(self._previous_state[1]) == 0:
            self._update_previous_state(frame)
            return 0., 0., 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, corner_found, err = cv2.calcOpticalFlowPyrLK(self._previous_state[0], gray, self._previous_state[1], None, **self._lk_params)

        good_new = p1[corner_found == 1].reshape(-1, 2)
        good_old = self._previous_state[1][corner_found == 1].reshape(-1, 2)
        self._update_previous_state(frame)
        if len(good_new) > 0:
            return tuple((good_new - good_old).mean(axis=0)) + (len(good_new),)
        else:
            return 0., 0., 0

    def _update_previous_state(self, frame):
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._previous_state = (frame, cv2.goodFeaturesToTrack(frame, mask=None, **self._feature_params))
