import numpy as np
from scipy import signal
from .core import AudioNode

class Filter(AudioNode):
    """
    Butterworth Filter (LowPass, HighPass, BandPass).
    """
    def __init__(self, filter_type: str = 'lowpass', cutoff: float = 1000.0, resonance: float = 0.707):
        """
        Args:
            filter_type: 'lowpass', 'highpass', or 'bandpass'
            cutoff: Cutoff frequency in Hz
            resonance: Q factor (default 0.707 for Butterworth)
        """
        self.filter_type = filter_type.lower()
        self.cutoff = cutoff
        self.resonance = resonance
        self.source = None

    def apply(self, source_node: AudioNode) -> 'Filter':
        self.source = source_node
        return self

    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        if self.source is None:
            raise ValueError("Filter needs a source node. Use .apply(node)")
        
        audio = self.source.get_samples(duration, rate)
        
        # Nyquist frequency
        nyq = 0.5 * rate
        # Normalized cutoff
        norm_cutoff = self.cutoff / nyq
        
        # Basic Butterworth filter
        # In V2, resonance (Q) could be used to adjust the filter order or character
        b, a = signal.butter(2, norm_cutoff, btype=self.filter_type)
        
        filtered = signal.lfilter(b, a, audio)
        return filtered.astype(np.float32)

    def __gt__(self, other):
        if isinstance(other, AudioNode):
            if hasattr(other, 'apply'):
                return other.apply(self)
        return super().__gt__(other)

class LowPassFilter(Filter):
    def __init__(self, cutoff=1000.0, resonance=0.707):
        super().__init__('lowpass', cutoff, resonance)

class HighPassFilter(Filter):
    def __init__(self, cutoff=1000.0, resonance=0.707):
        super().__init__('highpass', cutoff, resonance)

class PeakingFilter(AudioNode):
    """
    Parametric EQ (Peak/Notch).
    """
    def __init__(self, source: AudioNode = None, cutoff: float = 1000.0, gain_db: float = 0.0, Q: float = 1.0):
        self.source = source
        self.cutoff = cutoff
        self.gain_db = gain_db
        self.Q = Q

    def apply(self, source_node: AudioNode) -> 'PeakingFilter':
        self.source = source_node
        return self

    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        if self.source is None:
            raise ValueError("Filter needs a source node. Use .apply(node)")
            
        audio = self.source.get_samples(duration, rate)
        nyq = 0.5 * rate
        norm_cutoff = self.cutoff / nyq
        
        # Calculate iirpeak coefficients
        # Note: iirpeak implements a resonator. For a peaking EQ, we need to blend.
        # But as per request for surgical EQ "boost/cut", we'll use a simplified peaking approach.
        # scipy iirpeak is a notch/peak resonator.
        b, a = signal.iirpeak(norm_cutoff, self.Q)
        
        # Gain application
        gain = 10**(self.gain_db / 20.0)
        filtered = signal.lfilter(b, a, audio)
        
        # Peaking EQ result = Dry + Filtered * (Gain - 1)
        return (audio + filtered * (gain - 1.0)).astype(np.float32)
