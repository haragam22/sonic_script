"""
pysynth.instruments — Pre-made "Patches" (Drums, Bass, Synths)

Factory functions that return fully-wired AudioNode signal chains.
Each function accepts **kwargs so the transpiler's sequencer can inject
`freq` per step for pitched instruments.

PRD Section 5 defines the mathematical recipes:
  - Kick:   Sine sweep 150→40 Hz + fast envelope
  - Snare:  White noise + HPF + short decay
  - Hi-Hat: Metallic square wave + HPF + very short decay
  - Bass:   Saw oscillator + LPF + ADSR envelope
"""

import numpy as np
from .core import AudioNode
from .generators import Oscillator, NoiseNode, SineSweepNode, FMOscillator
from .envelopes import Envelope
from .filters import LowPassFilter, HighPassFilter, Filter


# ═══════════════════════════════════════════
#  DRUMS
# ═══════════════════════════════════════════

def kick(decay=0.2, **kwargs):
    """
    Kick drum — sine sweep from 150 Hz → 40 Hz with a fast envelope.
    
    PRD Recipe: "Sine wave sweeping from 150Hz → 0Hz in 0.1s + Fast Envelope."
    We sweep to 40 Hz (not 0) to keep audible sub-bass weight.
    """
    source = SineSweepNode(freq_start=150.0, freq_end=40.0, amp=1.0)
    env = Envelope(attack=0.001, decay=decay, sustain=0.0, release=0.05)
    return env.apply(source)


def snare(tone='bright', **kwargs):
    """
    Snare drum — white noise + high-pass filter + short decay.
    
    PRD Recipe: "White Noise + High-Pass Filter (1000Hz) + Short Decay."
    'tone' controls the HPF cutoff: 'bright' = 1000 Hz, 'dark' = 600 Hz.
    """
    cutoff = 1000.0 if tone == 'bright' else 600.0
    source = NoiseNode(amp=0.9)
    env = Envelope(attack=0.001, decay=0.15, sustain=0.0, release=0.1)
    hpf = HighPassFilter(cutoff=cutoff)
    return hpf.apply(env.apply(source))


def hi_hat(closed=True, **kwargs):
    """
    Hi-hat — metallic square wave + HPF + very short decay.
    
    PRD Recipe: "High-pitched square waves (metallic) + Band-Pass Filter."
    'closed' controls decay: True = tight 50ms, False = longer 200ms.
    """
    decay = 0.05 if closed else 0.2
    source = Oscillator(freq=6000.0, wave_type='square', amp=0.6)
    env = Envelope(attack=0.001, decay=decay, sustain=0.0, release=0.03)
    hpf = HighPassFilter(cutoff=5000.0)
    return hpf.apply(env.apply(source))


def clap(**kwargs):
    """
    Hand clap — noise burst through a bandpass filter with short decay.
    """
    source = NoiseNode(amp=0.8)
    env = Envelope(attack=0.001, decay=0.12, sustain=0.0, release=0.08)
    bpf = Filter(filter_type='bandpass', cutoff=1500.0)
    return bpf.apply(env.apply(source))


# ═══════════════════════════════════════════
#  BASS
# ═══════════════════════════════════════════

class BassNode(AudioNode):
    """
    Bass synthesizer — saw oscillator + low-pass filter + ADSR envelope.
    
    PRD Recipe: "Sawtooth wave + Low-Pass Filter (cutoff modulated by envelope)."
    """

    def __init__(self, freq=55.0, cutoff=300.0, amp=1.0, **kwargs):
        self.freq = freq
        self.cutoff = cutoff
        self.amp = amp

    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        osc = Oscillator(freq=self.freq, wave_type='saw', amp=self.amp)
        env = Envelope(attack=0.01, decay=0.3, sustain=0.6, release=0.2)
        lpf = LowPassFilter(cutoff=self.cutoff)
        node = lpf.apply(env.apply(osc))
        return node.get_samples(duration, rate)


def bass(freq=55.0, cutoff=300.0, **kwargs):
    """
    Convenience factory for BassNode.
    
    PRD: ps.Bass(note='C2') — accepts freq directly from the sequencer.
    """
    return BassNode(freq=freq, cutoff=cutoff, **kwargs)


# ═══════════════════════════════════════════
#  KEYS / SYNTHS
# ═══════════════════════════════════════════

def electric_piano(freq=440.0, **kwargs):
    """
    Electric piano — FM synthesis (bell-like Rhodes character).
    
    Uses 2-operator FM with moderate modulation index for that warm,
    bell-like attack that decays into a mellow sustain.
    """
    source = FMOscillator(
        freq=freq,
        carrier_type='sine',
        mod_ratio=2.0,
        mod_index=1.5,
        amp=0.7,
    )
    env = Envelope(attack=0.01, decay=0.4, sustain=0.3, release=0.3)
    return env.apply(source)


# ═══════════════════════════════════════════
#  REGISTRY — consumed by compiler/transpiler.py
# ═══════════════════════════════════════════

AVAILABLE_INSTRUMENTS = {
    # Drums
    'kick': kick,
    'snare': snare,
    'hi_hat': hi_hat,
    'clap': clap,
    # Bass
    'bass': bass,
    # Keys / Synths
    'electric_piano': electric_piano,
    'piano': electric_piano,        # alias
    'keys': electric_piano,         # alias (used in system_prompt examples)
}
