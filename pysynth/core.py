import numpy as np

class AudioNode:
    """Base class for all audio nodes in the signal chain."""
    
    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        """
        Generates audio samples for the given duration and sample rate.
        """
        raise NotImplementedError("Subclasses must implement get_samples")

    def apply(self, source_node: 'AudioNode') -> 'AudioNode':
        """Allows chaining nodes: effect.apply(source)"""
        if hasattr(self, 'source'):
            self.source = source_node
        return self

    def __add__(self, other):
        """Mixing two nodes by summation."""
        from .core import MixerNode 
        if not isinstance(other, AudioNode):
            raise TypeError("Can only mix nodes of type AudioNode")
        return MixerNode(self, other)

    def __mul__(self, volume):
        if isinstance(volume, (int, float)):
            return Gain(self, volume)
        raise TypeError("AudioNode can only be multiplied by a number (float/int)")

class MixerNode(AudioNode):
    """Internal node for mixing multiple sources."""
    def __init__(self, *sources):
        self.sources = list(sources)

    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        samples = [s.get_samples(duration, rate) for s in self.sources]
        mixed = np.sum(samples, axis=0)
        if len(samples) > 1:
            mixed = mixed / len(samples)
        return mixed.astype(np.float32)

class Gain(AudioNode):
    """Internal node for volume control."""
    def __init__(self, source, volume):
        self.source = source
        self.volume = volume

    def get_samples(self, duration, rate=44100):
        return self.source.get_samples(duration, rate) * self.volume