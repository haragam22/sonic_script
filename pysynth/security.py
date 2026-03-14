import RestrictedPython
from RestrictedPython import compile_restricted, safe_globals
import multiprocessing
import time
import sys

# Whitelisted modules
def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
    allowed_modules = ['pysynth', 'numpy', 'scipy', 'scipy.signal']
    if name in allowed_modules:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Module '{name}' is not allowed in the sandbox.")

def run_safe(code_str: str, duration: float = 1.0, rate: int = 44100):
    """
    Executes user code in a restricted environment with a timeout.
    
    Args:
        code_str: The Python code to execute.
        duration: Expected audio duration (passed to code).
        rate: Sample rate.
    """
    # Restricted environment setup
    # Note: RestrictedPython allows us to control access to built-ins
    
    # We'll use a queue to get the result from the subprocess
    result_queue = multiprocessing.Queue()

    def worker(q):
        try:
            # Prepare restricted globals
            loc = {}
            # We add our custom importer to the globals
            safe_globs = safe_globals.copy()
            safe_globs['__builtins__']['__import__'] = custom_import
            
            # Compile and execute
            byte_code = compile_restricted(code_str, filename='<string>', mode='exec')
            exec(byte_code, safe_globs, loc)
            
            # Expecting the code to define a variable 'audio'
            if 'audio' in loc:
                q.put(('success', loc['audio']))
            else:
                q.put(('error', "Variable 'audio' not found in generated code."))
        except Exception as e:
            q.put(('error', str(e)))

    # Start the process
    p = multiprocessing.Process(target=worker, args=(result_queue,))
    p.start()
    
    # Wait for result with timeout (2 seconds)
    p.join(timeout=2.0)
    
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError("Code execution exceeded 2-second timeout.")
    
    if result_queue.empty():
        raise RuntimeError("No result from sandbox process.")
    
    status, data = result_queue.get()
    if status == 'error':
        raise RuntimeError(f"Sandbox Escape or Error: {data}")
    
    return data

# Note: resource.setrlimit is not available on Windows.
# For memory capping on Windows, we'd typically use Job Objects or rely on the host system limits.
# In V1, we rely on the process isolation of multiprocessing.
