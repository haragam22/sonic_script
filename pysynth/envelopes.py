import numpy as np
from .core import AudioNode

class Envelope(AudioNode):
    """
    Standard ADSR Envelope (Attack, Decay, Sustain, Release).
    """
    def __init__(self, attack: float = 0.01, decay: float = 0.1, sustain: float = 0.5, release: float = 0.2):
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.source = None

    def apply(self, source_node: AudioNode) -> 'Envelope':
        """Chain a source node to this envelope."""
        self.source = source_node
        return self

    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        if self.source is None:
            raise ValueError("Envelope needs a source node. Use .apply(node)")
        
        audio = self.source.get_samples(duration, rate)
        n_samples = len(audio)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Calculate sample indices for stages
        a_end = int(self.attack * rate)
        d_end = a_end + int(self.decay * rate)
        r_start = n_samples - int(self.release * rate)
        
        envelope = np.ones(n_samples) * self.sustain
        
        # Attack stage: 0.0 to 1.0 linear ramp
        if a_end > 0:
            if a_end > n_samples: a_end = n_samples
            envelope[:a_end] = np.linspace(0, 1.0, a_end)
            
        # Decay stage: 1.0 to sustain level
        if d_end > a_end:
            if d_end > n_samples: d_end = n_samples
            if a_end < n_samples:
                envelope[a_end:d_end] = np.linspace(1.0, self.sustain, d_end - a_end)
                
        # Release stage: sustain level to 0.0
        if r_start < n_samples:
            if r_start < 0: r_start = 0
            # Ensure we don't overwrite attack/decay if they are long
            r_start = max(r_start, d_end)
            if r_start < n_samples:
                envelope[r_start:] = np.linspace(self.sustain, 0.0, n_samples - r_start)
        elif r_start >= n_samples and self.release > 0:
            # If duration is shorter than A+D+R, just fade out what's left
            pass

        return (audio * envelope).astype(np.float32)

    def __gt__(self, other):
        """Allow chaining like osc > envelope > filter"""
        if isinstance(other, AudioNode):
            if hasattr(other, 'apply'):
                return other.apply(self)
        return super().__gt__(other)
