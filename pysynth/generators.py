import numpy as np
from scipy import signal
from .core import AudioNode

class Oscillator(AudioNode):
    """
    Standard Oscillator supporting multiple wave types.
    """
    def __init__(self, freq: float = 440.0, wave_type: str = 'sine', amp: float = 1.0):
        self.freq = freq
        self.wave_type = wave_type
        self.amp = amp

    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        t = np.linspace(0, duration, int(rate * duration), endpoint=False)
        
        if self.wave_type == 'sine':
            audio = np.sin(2 * np.pi * self.freq * t)
        elif self.wave_type == 'square':
            audio = signal.square(2 * np.pi * self.freq * t)
        elif self.wave_type == 'saw':
            # scipy sawtooth defaults to rising, prd mentions "FT - floor(FT+0.5)" which is falling or standard sawtooth
            audio = signal.sawtooth(2 * np.pi * self.freq * t)
        elif self.wave_type == 'triangle':
            audio = signal.sawtooth(2 * np.pi * self.freq * t, width=0.5)
        else:
            raise ValueError(f"Unsupported wave_type: {self.wave_type}")
            
        return (audio * self.amp).astype(np.float32)

class NoiseNode(AudioNode):
    """
    White noise generator.
    """
    def __init__(self, amp: float = 1.0):
        self.amp = amp

    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        samples = np.random.uniform(-1, 1, int(rate * duration))
        return (samples * self.amp).astype(np.float32)

class SineSweepNode(AudioNode):
    """
    Sine wave with frequency sweep from start to end.
    """
    def __init__(self, freq_start: float = 150.0, freq_end: float = 40.0, amp: float = 1.0):
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.amp = amp

    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        t = np.linspace(0, duration, int(rate * duration), endpoint=False)
        # Linear frequency sweep
        freqs = np.linspace(self.freq_start, self.freq_end, len(t))
        phase = 2 * np.pi * np.cumsum(freqs) / rate
        audio = np.sin(phase)
        return (audio * self.amp).astype(np.float32)

class FMOscillator(AudioNode):
    """
    Frequency Modulation Oscillator.
    formula: sin(2*pi*f_c*t + I * sin(2*pi*f_m*t))
    where f_m = f_c * ratio
    """
    def __init__(self, freq: float = 440.0, carrier_type: str = 'sine', mod_ratio: float = 1.0, mod_index: float = 1.0, amp: float = 1.0):
        self.freq = freq
        self.carrier_type = carrier_type
        self.mod_ratio = mod_ratio
        self.mod_index = mod_index
        self.amp = amp

    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        t = np.linspace(0, duration, int(rate * duration), endpoint=False)
        
        # Modulator
        mod_freq = self.freq * self.mod_ratio
        modulator = self.mod_index * np.sin(2 * np.pi * mod_freq * t)
        
        # Carrier
        # The phase is 2*pi*f_c*t + modulator
        phase = 2 * np.pi * self.freq * t + modulator
        
        if self.carrier_type == 'sine':
            audio = np.sin(phase)
        elif self.carrier_type == 'square':
            audio = signal.square(phase)
        elif self.carrier_type == 'saw':
            audio = signal.sawtooth(phase)
        elif self.carrier_type == 'triangle':
            audio = signal.sawtooth(phase, width=0.5)
        else:
            raise ValueError(f"Unsupported carrier_type: {self.carrier_type}")
            
        return (audio * self.amp).astype(np.float32)

class LFO(Oscillator):
    """
    Low Frequency Oscillator for modulation.
    """
    def __init__(self, freq: float = 1.0, wave_type: str = 'sine', amp: float = 1.0):
        super().__init__(freq, wave_type, amp)

class Sampler(AudioNode):
    """
    File-based sample player.
    """
    def __init__(self, file_path: str, loop: bool = False, amp: float = 1.0):
        self.file_path = file_path
        self.loop = loop
        self.amp = amp
        self._data = None
        self._rate = None

    def _load(self):
        import soundfile as sf
        import os
        if not os.path.exists(self.file_path):
             # For demo purposes, create a dummy white noise file if not exists
             # Alternatively, handle the error gracefully.
             # In a real app, we'd raise FileNotFoundError.
             # Let's just raise it.
             raise FileNotFoundError(f"Sample file not found: {self.file_path}")
             
        self._data, self._rate = sf.read(self.file_path)
        # Ensure mono
        if len(self._data.shape) > 1:
            self._data = np.mean(self._data, axis=1)
        self._data = self._data.astype(np.float32)

    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        if self._data is None:
            self._load()
            
        n_samples = int(duration * rate)
        
        if self.loop:
            # Tiled looping
            audio = np.tile(self._data, int(n_samples / len(self._data)) + 1)[:n_samples]
        else:
            # Padded or truncated
            audio = self._data[:n_samples]
            if len(audio) < n_samples:
                audio = np.pad(audio, (0, n_samples - len(audio)))
                
        return (audio * self.amp).astype(np.float32)
