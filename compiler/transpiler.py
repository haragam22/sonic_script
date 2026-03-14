import inspect
import numpy as np
import random
from lark import Transformer, v_args
import pysynth as ps

# --- Robust Frequency Map ---
NOTE_FREQ = {
    "C": 16.35, "C#": 17.32, "Db": 17.32, "D": 18.35, "D#": 19.45, "Eb": 19.45, "E": 20.60,
    "F": 21.83, "F#": 23.12, "Gb": 23.12, "G": 24.50, "G#": 25.96, "Ab": 25.96, "A": 27.50,
    "A#": 29.14, "Bb": 29.14, "B": 30.87
}

def get_freq(note_str):
    """Safely converts a note string (C#2) to frequency (float)."""
    if not isinstance(note_str, str) or note_str in ('x', '-', '.'): 
        return 440.0
    
    # Handle simple regex parsing
    import re
    match = re.match(r"([A-G][b#]?)([0-9]+)", note_str)
    if not match: 
        return 440.0 # Default fallback
        
    note, octave = match.groups()
    base_freq = NOTE_FREQ.get(note, 27.50)
    return base_freq * (2 ** int(octave))

# --- Buffer Node for Sample Playback ---
class BufferNode(ps.AudioNode):
    """Wraps a raw numpy array into an AudioNode for chaining."""
    def __init__(self, data):
        self.data = data
    
    def get_samples(self, duration: float, rate: int = 44100) -> np.ndarray:
        req_samples = int(duration * rate)
        if self.data is None or len(self.data) == 0:
            return np.zeros(req_samples, dtype=np.float32)
        if req_samples <= len(self.data):
            return self.data[:req_samples]
        # Pad with silence if requested duration is longer than sample
        return np.pad(self.data, (0, req_samples - len(self.data)))

class InputNode(ps.AudioNode):
    """Placeholder for the master mix input. Can dynamically pull from transpiler tracks."""
    def __init__(self, data=None, transpiler=None):
        self.data = data
        self.transpiler = transpiler
        
    def get_samples(self, duration, rate=44100):
        req_samples = int(duration * rate)
        
        # 1. If we have explicit data (injected by runner.py), use it
        if self.data is not None:
            if req_samples <= len(self.data):
                return self.data[:req_samples]
            return np.pad(self.data, (0, req_samples - len(self.data)))
            
        # 2. Otherwise, dynamically mix current tracks in the transpiler
        if self.transpiler and self.transpiler.tracks:
            max_len = max(len(t) for t in self.transpiler.tracks.values())
            mix = np.zeros(max_len, dtype=np.float32)
            for buf in self.transpiler.tracks.values():
                mix[:len(buf)] += buf
            if len(self.transpiler.tracks) > 1:
                mix = mix / len(self.transpiler.tracks)
                
            if req_samples <= len(mix):
                return mix[:req_samples]
            return np.pad(mix, (0, req_samples - len(mix)))

        # 3. Fallback to silence
        return np.zeros(req_samples, dtype=np.float32)

# --- The Transpiler ---
class SonicTranspiler(Transformer):
    def __init__(self):
        super().__init__()
        self.env = {}
        self.tracks = {} # Stores rendered audio buffers for each track
        self.master_chain = None
        
        # Load Factory Instruments
        for name, func in ps.instruments.AVAILABLE_INSTRUMENTS.items():
            self.env[name] = func
            
        # Register Core Primitives (Aliases)
        self._register_core()

    def _register_core(self):
        """Robust mapping of core pysynth classes to DSL keywords."""
        # Generators
        self.env['saw']   = lambda **k: ps.Oscillator(wave_type='saw', **k)
        self.env['sine']  = lambda **k: ps.Oscillator(wave_type='sine', **k)
        self.env['sqr']   = lambda **k: ps.Oscillator(wave_type='square', **k)
        self.env['tri']   = lambda **k: ps.Oscillator(wave_type='triangle', **k)
        self.env['noise'] = ps.generators.NoiseNode
        self.env['fm']    = ps.generators.FMOscillator
        self.env['sweep'] = ps.generators.SineSweepNode
        self.env['lfo']   = ps.generators.LFO
        self.env['sampler'] = ps.generators.Sampler
        
        # Effects (Wrapped to handle positional args correctly in chains)
        self.env['lpf'] = lambda cutoff=1000, **k: ps.LowPassFilter(cutoff=cutoff, **k)
        self.env['hpf'] = lambda cutoff=1000, **k: ps.HighPassFilter(cutoff=cutoff, **k)
        self.env['peaking'] = lambda cutoff=1000, gain_db=0.0, **k: ps.PeakingFilter(cutoff=cutoff, gain_db=gain_db, **k)
        self.env['reverb'] = lambda mix=0.3, **k: ps.Reverb(mix=mix, **k)
        self.env['delay'] = lambda mix=0.5, **k: ps.Delay(mix=mix, **k)
        self.env['maximizer'] = lambda amount=0.5, **k: ps.MaximizerNode(amount=amount, **k)
        self.env['dist'] = self.env['maximizer']
        self.env['drive'] = self.env['maximizer']
        self.env['sidechain'] = ps.Sidechain
        self.env['timewarp'] = ps.TimeWarp
        self.env['gain'] = lambda volume=1.0, **k: ps.Gain(source=None, volume=volume, **k)
        self.env['granulate'] = ps.Granulate
        self.env['glitch'] = ps.Granulate
        self.env['sliceloop'] = ps.SliceLoop
        self.env['decay'] = lambda d=0.1, **k: ps.Envelope(attack=0.001, decay=d, sustain=0.0, release=d, **k)
        
        # Robust Vibe Check (Handles 'audio', 'input', 'prompt' etc.)
        def safe_vibe(**k):
            # Map common AI-isms to actual params
            audio = k.get('audio') or k.get('input') or k.get('audio_input')
            prompt = k.get('prompt', "")
            return ps.check_vibe(audio=audio, prompt=prompt)
        
        self.env['check_vibe'] = safe_vibe
        self.env['mix'] = lambda **k: ps.InputNode(**k) # Alias for input

    # --- Utilities ---
    def _bind_args(self, func, pos_args, kw_args):
        """
        Uses introspection to bind DSL arguments, filtering out 
        hallucinated keywords that would cause TypeErrors.
        """
        try:
            sig = inspect.signature(func)
            
            # 1. Map positional args to their names
            parameters = list(sig.parameters.values())
            bound_kw = kw_args.copy()
            
            for i, val in enumerate(pos_args):
                if i < len(parameters):
                    name = parameters[i].name
                    if name not in bound_kw:
                        bound_kw[name] = val
            
            # 2. Filter keywords by signature
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters)
            if not has_kwargs:
                final_kw = {k: v for k, v in bound_kw.items() if k in sig.parameters}
            else:
                final_kw = bound_kw
                
            # 3. Bind
            bound = sig.bind_partial(**final_kw)
            return [], bound.kwargs # Convert all to keyword for safety
        except Exception:
            # Fallback to raw if logic fails
            return pos_args, kw_args

    # --- Transformer Methods ---
    
    def start(self, items):
        # Return the final state for the Runner
        return {
            "tracks": self.tracks,
            "master_chain": self.master_chain
        }

    def assign_stmt(self, items):
        name = str(items[0])
        value = items[1]
        self.env[name] = value
        return None

    def chain(self, items):
        """
        Handles signal flow: Source | Effect | Effect
        """
        source = items[0]
        effects = items[1:]
        
        # 1. Handle Function Templates (Lazy Evaluation)
        if callable(source) and not isinstance(source, ps.AudioNode):
            def template(**kwargs):
                # Instantiate source
                pos, kw = self._bind_args(source, [], kwargs)
                node = source(*pos, **kw)
                # Apply effects
                for effect in effects:
                    node = effect.apply(node)
                return node
            return template
            
        # 2. Handle Static Nodes (Immediate Evaluation)
        current_node = source
        for effect in effects:
            if hasattr(effect, 'apply'):
                current_node = effect.apply(current_node)
            else:
                raise ValueError(f"Object {effect} is not an effect (missing .apply())")
        return current_node

    def import_stmt(self, items):
        return None

    def mix_function(self, items):
        # If any item is a template, return a template
        if any(callable(i) and not isinstance(i, ps.AudioNode) for i in items):
            def template(**kwargs):
                resolved = []
                for i in items:
                    if callable(i) and not isinstance(i, ps.AudioNode):
                        # Ensure we don't pass extra args if not accepted
                        pos, kw = self._bind_args(i, [], kwargs)
                        resolved.append(i(*pos, **kw))
                    else:
                        resolved.append(i)
                return ps.MixerNode(*resolved)
            return template
        return ps.MixerNode(*items)

    def instrument_call(self, items):
        name = str(items[0])
        args_tuple = items[1] if len(items) > 1 else ([], {})
        pos_args, kw_args = args_tuple

        target = self.env.get(name)
        if not target:
            raise ValueError(f"Unknown instrument: '{name}'")

        try:
            # Case A: It's a Class (e.g. LowPassFilter) -> Instantiate it
            if inspect.isclass(target):
                pos, kw = self._bind_args(target, pos_args, kw_args)
                return target(*pos, **kw)
                
            # Case B: It's a Function (e.g. kick()) -> Call it
            if callable(target):
                pos, kw = self._bind_args(target, pos_args, kw_args)
                return target(*pos, **kw)
        except TypeError as e:
            raise ValueError(f"Argument Mismatch in '{name}': {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error calling '{name}': {str(e)}")
            
        # Case C: It's a variable reference
        return target

    def track_block(self, items):
        track_name = str(items[0])
        bpm = 120
        swing = 0.0
        humanize = 0.0
        plays = []
        
        # Extract content
        for item in items[1:]:
            if isinstance(item, tuple):
                if item[0] == 'bpm': bpm = item[1]
                elif item[0] == 'swing': swing = item[1]
                elif item[0] == 'humanize': humanize = item[1]
                elif item[0] == 'play': plays.append(item[1])

        # --- Sequencer Logic ---
        rate = 44100
        beat_len = 60 / bpm
        grid_step = beat_len / 4 # 16th notes
        
        # Determine track length
        max_steps = 0
        for _, pattern in plays:
            max_steps = max(max_steps, len(pattern))
            
        total_samples = int(max_steps * grid_step * rate) + rate # +1s buffer
        buffer = np.zeros(total_samples, dtype=np.float32)

        try:
            for instr_name, pattern in plays:
                instr_func = self.env.get(instr_name)
                if not instr_func:
                    print(f"Warning: Instrument '{instr_name}' not found. Skipping.")
                    continue

                for step, note in enumerate(pattern):
                    if note in ('-', '.'): continue
                    
                    # Dynamic instantiation based on note
                    if callable(instr_func):
                        # Robustness: Check if it accepts 'freq'
                        sig = inspect.signature(instr_func)
                        
                        if 'freq' in sig.parameters and note != 'x':
                            node = instr_func(freq=get_freq(note))
                        elif 'note' in sig.parameters:
                            node = instr_func(note=note)
                        else:
                            node = instr_func() # Fallback (e.g. drum hit)
                    else:
                        node = instr_func # Static node reference

                    # Render sample (16th note length)
                    # Note: We let it ring out slightly (grid_step * 2) for release tails
                    samples = node.get_samples(grid_step * 2.0, rate)
                    
                    # Mix into buffer with SWING and HUMANIZE
                    # If step is odd (1, 3, 5...), delay it
                    swing_offset = 0
                    if step % 2 == 1:
                        swing_offset = swing * (grid_step / 2)

                    # Add random jitter (Humanize)
                    jitter = random.uniform(-humanize, humanize)
                    
                    exact_time = step * grid_step + swing_offset + jitter
                    start_time = max(0, exact_time)
                    
                    start_idx = int(start_time * rate)
                    end_idx = start_idx + len(samples)
                    
                    # Apply velocity if specified
                    velocity = 1.0
                    if isinstance(note, tuple) and note[0] == 'vel':
                        velocity = note[1]
                        samples = samples * velocity

                    if start_idx < len(buffer):
                        actual_end = min(end_idx, len(buffer))
                        buffer[start_idx:actual_end] += samples[:actual_end-start_idx]
        except Exception as e:
            raise RuntimeError(f"Synthesis Error in track '{track_name}': {str(e)}")

        self.tracks[track_name] = buffer
        self.env[track_name] = BufferNode(buffer)
        return None

    def master_item(self, items):
        if items and items[0] is not None:
             self.master_chain = items[0]
        return None

    def master_block(self, items):
        return None

    # --- Argument Helpers ---
    def argument_list(self, items):
        pos = []
        kw = {}
        for item in items:
            if isinstance(item, dict): kw.update(item)
            else: pos.append(item)
        return pos, kw

    def positional_arg(self, items): return items[0]
    def keyword_arg(self, items): return {str(items[0]): items[1]}
    def mix_arg(self, items): return {"mix": items[0]}
    
    # --- Primitives ---
    def number(self, items): return float(items[0])
    def string(self, items): return items[0][1:-1]
    def var_ref(self, items): 
        name = str(items[0])
        if name not in self.env: raise ValueError(f"Undefined variable: {name}")
        return self.env[name]
    
    # Special: The 'input' keyword for Master Block
    def input_ref(self, items):
        return InputNode(transpiler=self)
    
    # --- Sequencer Primitives ---
    def pattern(self, items): return items
    def note(self, items): return str(items[0])
    def accent(self, items): return ('vel', 1.2)
    def trigger(self, items): return ('vel', 0.9)
    def ghost(self, items): return ('vel', 0.4)
    def rest(self, items): return '-'
    def silence(self, items): return '.'

    def euclidean_call(self, items):
        hits = int(items[0])
        steps = int(items[1])
        # Bjorklund-style simple linear distribution
        res = [1 if (i * hits) % steps < hits else 0 for i in range(steps)]
        return [('vel', 0.9) if r == 1 else '-' for r in res]
    
    def top_level_call(self, items):
        # We actually execute the call here (e.g. check_vibe)
        # This allows the AI to trigger vibe checks with specific prompts
        return items[0] 
    
    # --- Stubs ---
    def bpm_stmt(self, items):
        try:
            value = float(items[0])
        except (ValueError, TypeError):
             raise ValueError(f"BPM must be a number, got '{items[0]}'")
             
        if value <= 0 or value > 1000:
             raise ValueError(f"BPM {value} is out of realistic range (1-1000)")
        return ('bpm', value)

    def swing_stmt(self, items):
        try:
            value = float(items[0])
            return ('swing', value)
        except (ValueError, TypeError):
            raise ValueError(f"Swing must be a number, got '{items[0]}'")

    def humanize_stmt(self, items):
        try:
            value = float(items[0])
            return ('humanize', value)
        except (ValueError, TypeError):
            raise ValueError(f"Humanize must be a number, got '{items[0]}'")

    def play_stmt(self, items): return ('play', (str(items[0]), items[1]))
    def ignore_comment(self, items): return None
    def ignore_newline(self, items): return None