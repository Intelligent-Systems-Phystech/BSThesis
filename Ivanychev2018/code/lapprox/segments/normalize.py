from typing import Iterable, List, Optional

import numpy as np
import scipy


def shrink_segment(segment: np.ndarray, output_length: int) -> np.ndarray:
    segment_indices = np.arange(segment.size)
    interpolated_f = scipy.interpolate.interp1d(segment_indices,
                                                segment,
                                                kind='cubic')
    new_indices = np.linspace(0, segment_indices[-1], output_length)
    return interpolated_f(new_indices)


def normalize_segments(segments: Iterable[np.ndarray],
                       length: Optional[int]=None) -> List[np.ndarray]:
    segments = list(segments)
    length = length if length else min(segment.size for segment in segments)
    return [shrink_segment(segment, length) for segment in segments]
