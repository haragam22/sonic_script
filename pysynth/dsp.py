import numpy as np
import random
from .core import AudioNode

class Granulate(AudioNode):
    def __init__(self, source: AudioNode = None, grain_size=0.1, density=2.0, scatter=0.05):
        """
        :param source: The audio to granulate.
        :param grain_size: Duration of each grain in seconds (0.01 to 0.5).
        :param density: Overlap factor (2.0 = 2 grains playing at once).
        :param scatter: Randomness in playback position (0.0 to 1.0).
        """
        self.source = source
        self.grain_size = grain_size
        self.density = density
        self.scatter = scatter

    def apply(self, source_node: AudioNode) -> 'Granulate':
        self.source = source_node
        return self

    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        if self.source is None:
            raise ValueError("Granulate needs a source node. Use .apply(node)")
            
        # 1. Render the source audio first
        src_audio = self.source.get_samples(duration, rate)
        output = np.zeros_like(src_audio)
        
        # 2. Calculate Grain Parameters
        grain_len_samples = int(self.grain_size * rate)
        if grain_len_samples < 1: grain_len_samples = 1
        
        step_size = int(grain_len_samples / self.density)
        if step_size < 1: step_size = 1
        
        # Hanning window to smooth edges of grains
        window = np.hanning(grain_len_samples)
        
        # 3. The Granulation Loop
        for write_pos in range(0, len(output) - grain_len_samples, step_size):
            # Calculate "Read Head" with randomness (Scatter)
            jitter = int((random.random() - 0.5) * 2 * self.scatter * rate)
            read_pos = write_pos + jitter
            
            # Bounds check
            if read_pos < 0: read_pos = 0
            if read_pos + grain_len_samples >= len(src_audio): 
                read_pos = len(src_audio) - grain_len_samples - 1
            
            if read_pos < 0: continue

            # Extract and Window the Grain
            grain = src_audio[read_pos : read_pos + grain_len_samples] * window
            
            # Overlap-Add to output
            output[write_pos : write_pos + grain_len_samples] += grain

        return output.astype(np.float32)

class Sidechain(AudioNode):
    """
    Compresses source volume based on trigger signal amplitude.
    """
    def __init__(self, source: AudioNode = None, trigger: AudioNode = None, threshold: float = 0.5, ratio: float = 4.0):
        self.source = source
        self.trigger = trigger
        self.threshold = threshold
        self.ratio = ratio

    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        src_audio = self.source.get_samples(duration, rate)
        trig_audio = self.trigger.get_samples(duration, rate)
        
        # Calculate envelope of trigger (absolute value)
        envelope = np.abs(trig_audio)
        
        # Compression logic
        gain = np.ones_like(envelope)
        mask = envelope > self.threshold
        
        # Simple ratio: gain = 1 - (1 - 1/ratio) * (env - thresh) / (1 - thresh)
        # We'll use a slightly simpler linear gain reduction for V1
        reduction = (envelope[mask] - self.threshold) * (1 - 1/self.ratio)
        gain[mask] -= reduction
        
        # Ensure gain doesn't go negative
        gain = np.maximum(gain, 0.0)
        
        return (src_audio * gain).astype(np.float32)

def granulate(source_audio: np.ndarray, rate: int = 44100, grain_size: float = 0.05, density: float = 0.8, scatter: float = 0.2) -> np.ndarray:
    """
    Windowed grain shuffling.
    """
    n_samples = len(source_audio)
    grain_len = int(grain_size * rate)
    if grain_len < 1: grain_len = 1
    
    output = np.zeros_like(source_audio)
    step = int(grain_len * (1.0 - density))
    if step < 1: step = 1
    
    for i in range(0, n_samples - grain_len, step):
        # Determine source position with scatter
        pos = i + int((np.random.rand() - 0.5) * scatter * rate)
        pos = np.clip(pos, 0, n_samples - grain_len)
        
        grain = source_audio[pos:pos+grain_len]
        
        # Window the grain (Hanning)
        window = np.hanning(grain_len)
        output[i:i+grain_len] += grain * window
        
    return output.astype(np.float32)

class TimeWarp(AudioNode):
    """
    Circular buffer manipulation for reverse/half-speed effects.
    """
    def __init__(self, source: AudioNode = None, grid: str = '1/4', pattern: str = 'reverse'):
        self.source = source
        self.grid = grid # e.g., '1/4'
        self.pattern = pattern

    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        audio = self.source.get_samples(duration, rate)
        # For V1, we'll reverse the entire array if pattern is 'reverse'
        # Proper grid-based slicing is for V2
        if self.pattern == 'reverse':
            return np.flip(audio).astype(np.float32)
        elif self.pattern == 'half-speed':
            # Linear interpolation for speed change
            from scipy import interpolate
            x = np.linspace(0, 1, len(audio))
            f = interpolate.interp1d(x, audio)
            new_x = np.linspace(0, 0.5, len(audio)) # Only take first half of time
            return f(new_x).astype(np.float32)
        return audio

def maximizer(audio: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """
    Soft-clipping using tanh for loudness.
    """
    # Drive the signal
    gain = 1.0 + amount * 5.0
    driven = audio * gain
    # Soft clip
    clipped = np.tanh(driven)
    return clipped.astype(np.float32)

class MaximizerNode(AudioNode):
    def __init__(self, source: AudioNode = None, amount: float = 0.5):
        self.source = source
        self.amount = amount
    
    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        audio = self.source.get_samples(duration, rate)
        return maximizer(audio, self.amount)

class Delay(AudioNode):
    """
    Feedback Delay Effect.
    """
    def __init__(self, source: AudioNode = None, time: float = 0.5, feedback: float = 0.4, mix: float = 0.5):
        self.source = source
        self.time = time
        self.feedback = feedback
        self.mix = mix

    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        dry = self.source.get_samples(duration, rate)
        delay_samples = int(self.time * rate)
        
        if delay_samples <= 0:
            return dry
            
        wet = np.zeros_like(dry)
        # Simple iterative feedback delay in chunks isn't efficient in Python
        # But for V1 we do a single-pass buffer mix
        for i in range(delay_samples, len(dry)):
            wet[i] = dry[i - delay_samples] + wet[i - delay_samples] * self.feedback
            
        return (dry * (1 - self.mix) + wet * self.mix).astype(np.float32)

class Reverb(AudioNode):
    """
    Schroeder Reverb (Freeverb style approximation).
    4 Parallel Comb Filters -> 2 Series All-pass Filters.
    """
    def __init__(self, source: AudioNode = None, room_size: float = 0.5, damp: float = 0.5, mix: float = 0.3):
        self.source = source
        self.room_size = room_size # Controls feedback of comb filters
        self.damp = damp           # Controls low-pass in comb filters
        self.mix = mix

    def _comb_filter(self, audio, delay, feedback, damp):
        output = np.zeros_like(audio)
        last_out = 0
        for i in range(delay, len(audio)):
            # Feedback block with LPF (damping)
            val = audio[i - delay] + last_out * feedback
            output[i] = val
            last_out = val * (1 - damp)
        return output

    def _allpass_filter(self, audio, delay, feedback):
        output = np.zeros_like(audio)
        for i in range(delay, len(audio)):
            # y[n] = -g * x[n] + x[n-d] + g * y[n-d]
            output[i] = -feedback * audio[i] + audio[i - delay] + feedback * output[i - delay]
        return output

    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        dry = self.source.get_samples(duration, rate)
        
        # Parallel Comb Filters (different delay times in samples)
        # Traditional Schroeder values: 1116, 1188, 1277, 1356
        c1 = self._comb_filter(dry, 1116, 0.7 + self.room_size * 0.28, self.damp * 0.4)
        c2 = self._comb_filter(dry, 1188, 0.7 + self.room_size * 0.28, self.damp * 0.4)
        c3 = self._comb_filter(dry, 1277, 0.7 + self.room_size * 0.28, self.damp * 0.4)
        c4 = self._comb_filter(dry, 1356, 0.7 + self.room_size * 0.28, self.damp * 0.4)
        
        wet = (c1 + c2 + c3 + c4) * 0.25
        
        # Series All-pass Filters
        wet = self._allpass_filter(wet, 556, 0.5)
        wet = self._allpass_filter(wet, 441, 0.5)
        
        return (dry * (1 - self.mix) + wet * self.mix).astype(np.float32)
