import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from .parser import get_parser
from .transpiler import SonicTranspiler
from .runner import run_ast

# Define version here
VERSION = "0.2.1-alpha"

def plot_mix(audio_data, rate=44100):
    """Generates a Pop-Up Window with Waveform & Spectrum"""
    print("--> Generating Mix Report...")
    
    # Create Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.canvas.manager.set_window_title('White-Box Composer: Mix Report')
    
    # 1. Waveform (Time Domain)
    time = np.linspace(0, len(audio_data) / rate, num=len(audio_data))
    ax1.plot(time, audio_data, color='#00ff00', linewidth=0.5)
    ax1.set_title("Waveform (Dynamics)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5) # Clipping line
    ax1.axhline(y=-1.0, color='r', linestyle='--', alpha=0.5)

    # 2. Spectrogram (Frequency Domain)
    ax2.specgram(audio_data, NFFT=1024, Fs=rate, noverlap=512, cmap='inferno')
    ax2.set_title("Spectrogram (Frequency Balance)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_yscale('linear') 
    ax2.set_ylim(0, 15000) # Focus on audible range

    plt.tight_layout()
    plt.show() # This pauses execution until window is closed

def run_file(filepath, visualize=False):
    """The core logic to run a .sonic file"""
    if not os.path.exists(filepath):
        print(f"Runtime Error: File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()

        # 1. Parse
        parser_engine = get_parser()
        print(f"--> Parsing '{filepath}'...")
        tree = parser_engine.parse(code)
        
        # 2. Transpile & Execute
        final_audio = run_ast(tree) 
        
        # 3. Visualization
        if visualize and final_audio is not None:
            plot_mix(final_audio)
            
    except Exception as e:
        # Print neat error to stderr so conductor.py can catch it
        print(f"Runtime Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        prog="sonic",
        description="White-Box Composer Compiler (SonicScript)",
        epilog="Treat music as code."
    )
    
    # 1. The File Argument (Positional)
    parser.add_argument("file", nargs="?", help="The .sonic file to compile")
    
    # 2. Flags
    parser.add_argument("-v", "--visualize", action="store_true", help="Show real-time spectrogram")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    
    args = parser.parse_args()

    # Logic: If no file is provided, show help
    if not args.file:
        parser.print_help()
        sys.exit(0)
        
    # Logic: Run the file
    run_file(args.file, args.visualize)

if __name__ == "__main__":
    main()
