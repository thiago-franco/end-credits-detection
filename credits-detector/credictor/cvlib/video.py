import cv2
from typing import Union, Optional

import numpy as np

from credictor.cvlib.image import Image


class VideoProgress:
    def init(self):
        raise NotImplementedError()

    def update(self, n: int):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class NoVideoProgress(VideoProgress):
    def init(self):
        pass

    def update(self, n: int):
        pass

    def close(self):
        pass


class TqdmVideoProgress(VideoProgress):
    def __init__(self):
        try:
            import tqdm
        except ImportError:
            raise ImportError("TqdmVideoProgress needs tqdm to be installed")
        self._tqdm = tqdm
        self._progress_bar = None  # type: tqdm.tqdm

    def init(self):
        self._progress_bar = self._tqdm.tqdm(unit='frame')

    def update(self, n: int):
        self._progress_bar.update(n)

    def close(self):
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None


class Video(object):
    def __init__(self, src: Union[int, str], read_every: int = 1, start_at: float = 0.0,
                 progress: Union[bool, VideoProgress] = False):
        self._src = src
        self._read_every = read_every
        self._start_at = start_at
        self._cap = None  # type: Optional[cv2.VideoCapture]
        self._frame = None  # type: np.ndarray
        self._ret = False  # type: bool
        self._progress = _get_progress_object(progress)
        self.init()

    def init(self) -> None:
        self.close()
        self._cap = cv2.VideoCapture(self._src)
        if not self._cap.isOpened():
            raise ValueError("Couldn't open cap for '%s'" % str(self._src))
        self._progress.init()
        self.crop()
        
    def crop(self) -> None:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES,int(self.length() * self._start_at))
        self._grab_frame()    

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._progress.close()

    def has_next(self) -> bool:
        return self._ret

    def next(self) -> Image:
        frame = self.peek()
        self._grab_frame()
        return frame

    def peek(self) -> Image:
        if not self._ret:
            raise ValueError("There are no frames left")
        return Image(self._frame)

    def _grab_frame(self):
        for i in range(self._read_every):
            self._ret, self._frame = self._cap.read()
        self._progress.update(self._read_every)

    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)
    
    def length(self) -> int:
        return self._cap.get(cv2.CAP_PROP_FRAME_COUNT)

    @property
    def frame_reading_window(self) -> int:
        return self._read_every
    
    @property
    def cropped_proportion(self) -> float:
        return self._start_at

    def __iter__(self):
        self.init()
        return self

    def __next__(self):
        if self.has_next():
            return self.next()
        else:
            self.close()
            raise StopIteration

    def __del__(self):
        self.close()


def _get_progress_object(progress: Union[bool, VideoProgress]) -> VideoProgress:
    if progress is True:
        return TqdmVideoProgress()
    elif progress is False:
        return NoVideoProgress()
    elif isinstance(progress, VideoProgress):
        return progress
    else:
        raise ValueError('`progress` should be a boolean or a VideoProgress object')
