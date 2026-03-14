import numpy as np
from scipy.signal import find_peaks
from .core import AudioNode

class BufferNode(AudioNode):
    """Simple node that holds a static audio buffer."""
    def __init__(self, data):
        self.data = data
    def get_samples(self, duration, rate=44100):
        # Return slice of data matching duration
        req_samples = int(duration * rate)
        if req_samples > len(self.data):
            # Pad with silence if requested duration is longer than slice
            return np.pad(self.data, (0, req_samples - len(self.data)))
        return self.data[:req_samples]

def SliceLoop(node: AudioNode, threshold: float = 0.1, duration: float = 4.0, rate: int = 44100):
    """
    Detects transients and returns a list of BufferNodes (slices).
    :param node: The drum loop to slice.
    :param threshold: Sensitivity (0.0 to 1.0). Lower = more slices.
    :param duration: How much of the source to analyze.
    """
    # 1. Render the full loop to analyze it
    audio = node.get_samples(duration, rate)
    
    # 2. Transient Detection Algorithm
    # Step A: Get Amplitude Envelope (Absolute value)
    envelope = np.abs(audio)
    
    # Step B: Calculate "Energy Difference" (How fast volume is rising)
    # We put a 0 at the start to keep array length same
    energy_diff = np.diff(envelope, prepend=0)
    
    # Step C: Find Peaks in the difference (Sudden volume spikes)
    # distance=int(rate * 0.05) means "don't find two hits within 50ms of each other"
    peaks, _ = find_peaks(energy_diff, height=threshold, distance=int(rate * 0.05))
    
    # 3. Chop the Audio
    slices = []
    start_idx = 0
    
    for peak_idx in peaks:
        # Create a slice from previous start to this peak
        if peak_idx > start_idx:
            chunk = audio[start_idx:peak_idx]
            slices.append(BufferNode(chunk))
        
        start_idx = peak_idx
        
    # Add the final chunk
    if start_idx < len(audio):
        slices.append(BufferNode(audio[start_idx:]))
    
    return slices
