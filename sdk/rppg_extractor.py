import numpy as np
import pywt
from scipy.signal import butter, filtfilt
from typing import List


def bandpass_filter(signal: np.ndarray, fs: float = 30.0, low_hz: float = 0.7, high_hz: float = 4.0):
    if len(signal) < 21: # Min length for 3rd order filtfilt and padding
        return signal
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = butter(3, [low, high], btype="band")
    return filtfilt(b, a, signal)


class RPPGExtractor:
    """CHROM-based rPPG extractor operating on a sequence of forehead ROIs.

    Usage: collect N frames of cropped ROI (H x W x 3), build a buffer and call
    `extract` to get a 1-D pulse signal aligned with frame rate.
    """

    def __init__(self, fs: float = 30.0):
        self.fs = fs

    def _rgb_means(self, roi_buffer: List[np.ndarray]):
        # roi_buffer: list of cropped BGR frames
        arr = np.stack([cv2_mean_rgb(a) for a in roi_buffer], axis=0)
        return arr

    def extract(self, roi_buffer: List[np.ndarray], rgb_means: np.ndarray = None) -> np.ndarray:
        """
        Extract the raw rPPG signal (CHROM) and apply Wavelet Denoising.
        Accepts either a list of cropped BGR frames (roi_buffer) OR pre-computed rgb_means directly.
        """
        try:
            if rgb_means is not None:
                arr = rgb_means
            else:
                if len(roi_buffer) < 10:
                    return np.array([])
                arr = self._rgb_means(roi_buffer)

            if len(arr) < 10:
                return np.array([])

            r = arr[:, 2]
            g = arr[:, 1]
            b = arr[:, 0]

            # CHROM method transforms
            X = 3 * r - 2 * g
            Y = 1.5 * r + g - 1.5 * b
            
            # Standardize components for local stability
            X = (X - np.mean(X)) / (np.std(X) + 1e-12)
            Y = (Y - np.mean(Y)) / (np.std(Y) + 1e-12)
            
            # CHROM projection
            S = X - Y

            # Wavelet Denoising (Discrete Wavelet Transform)
            # Using 'db4' Daubechies wavelet for pulse-like feature preservation
            coeffs = pywt.wavedec(S, 'db4', level=min(3, pywt.dwt_max_level(len(S), 'db4')))
            # Soft-threshold high-frequency detail coefficients
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(S)))
            coeffs[1:] = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[1:]]
            S_denoised = pywt.waverec(coeffs, 'db4')
            # Ensure length matches original projection
            S_denoised = S_denoised[:len(S)]

            # SpO2 Ratios (AC/DC)
            # Use standard deviation as AC proxy and mean as DC proxy
            r_ac = np.std(r)
            r_dc = np.mean(r) + 1e-12
            b_ac = np.std(b)
            b_dc = np.mean(b) + 1e-12
            
            r_ratio = r_ac / r_dc
            b_ratio = b_ac / b_dc

            # Detrend and filter
            S_filtered = bandpass_filter(S_denoised, fs=self.fs)
            
            return {
                "pulse_signal": S_filtered,
                "r_ratio": float(r_ratio),
                "b_ratio": float(b_ratio)
            }
        except Exception as e:
            print(f"[NeuroVitals] [RPPG_EXTRACTOR] ERROR in extract: {e}")
            return {
                "pulse_signal": np.array([]),
                "r_ratio": 1.0,
                "b_ratio": 1.0
            }


def cv2_mean_rgb(frame: np.ndarray):
    # compute mean color inside non-zero area
    import numpy as _np

    if frame is None or frame.size == 0:
        return _np.array([0.0, 0.0, 0.0])
    mask = _np.any(frame != 0, axis=2)
    if not mask.any():
        return _np.array([0.0, 0.0, 0.0])
    vals = frame[mask]
    return _np.mean(vals, axis=0)
