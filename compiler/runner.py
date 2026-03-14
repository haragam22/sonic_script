import numpy as np
import sounddevice as sd
import pysynth as ps
from .transpiler import SonicTranspiler, InputNode, BufferNode

def run_ast(tree):
    """
    Transpiles the AST and executes the resulting audio graph.
    """
    print("--> Transpiling SonicScript...")
    transpiler = SonicTranspiler()
    results = transpiler.transform(tree)
    
    tracks_dict = results.get('tracks', {})
    master_chain = results.get('master_chain')
    
    if not tracks_dict:
        print("Warning: No tracks found to play.")
        return

    # 1. Mix all tracks for the 'input' bus
    max_samples = max(len(buf) for buf in tracks_dict.values())
    summed_audio = np.zeros(max_samples, dtype=np.float32)
    
    for name, buf in tracks_dict.items():
        max_track = np.max(np.abs(buf))
        print(f"  - Mixing track: {name} (Max Amp: {max_track:.4f})")
        summed_audio[:len(buf)] += buf
        
    # Standard normalization for the raw mix
    if len(tracks_dict) > 1:
        summed_audio = summed_audio / len(tracks_dict)
    
    print(f"--> Raw Mix Max Amp: {np.max(np.abs(summed_audio)):.4f}")

    # 2. Final Audio Node Selection
    final_node = None
    
    if master_chain:
        print("--> Applying Master Effects...")
        
        # Scenario A: Master Chain is a Template (Callable but not a Node)
        if callable(master_chain) and not isinstance(master_chain, ps.AudioNode):
            mix_node = BufferNode(summed_audio)
            final_node = master_chain(source=mix_node)
        else:
            # Scenario B: Master Chain is an active node graph (e.g. input | effect)
            # Find the InputNode placeholder and inject the summed audio
            def find_and_inject(node, data):
                if isinstance(node, InputNode):
                    node.data = data
                    return True
                # Recurse through parents
                found = False
                if hasattr(node, 'source') and node.source:
                    if find_and_inject(node.source, data): found = True
                if hasattr(node, 'sources') and node.sources: # MixerNodes
                    for s in node.sources:
                        if find_and_inject(s, data): found = True
                return found

            if find_and_inject(master_chain, summed_audio):
                final_node = master_chain
            else:
                # If no 'input' was used, the master_chain is the final source itself
                # (e.g. master: mix [t1, t2])
                final_node = master_chain
    else:
        # No master block defined, just use the raw mix
        final_node = BufferNode(summed_audio)

    # 3. Render Final Audio
    duration = len(summed_audio) / 44100
    final_audio = final_node.get_samples(duration)

    # 4. Final Normalization & Safety
    max_val = np.max(np.abs(final_audio))
    if max_val > 0.0001:
        # Prevent clipping while keeping impact
        final_audio = final_audio / max_val
    
    # Smooth fade out to prevent clicks at end
    fade_len = int(0.05 * 44100)
    if len(final_audio) > fade_len:
        fade = np.linspace(1.0, 0.0, fade_len)
        final_audio[-fade_len:] *= fade

    # 5. Vibe Check
    try:
        ps.check_vibe(final_audio, "SonicScript Composition")
        print("--> Vibe Check: PASSED")
    except Exception as e:
        print(f"--> Vibe Check Result: {e}")

    # 6. Playback
    print("--> Playing...")
    sd.play(final_audio, 44100)
    sd.wait()
    print("--> Done.")
    return final_audio
