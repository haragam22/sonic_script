import os
import subprocess
import sys
import time
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Initialize Terminal Colors
init(autoreset=True)

class AIBrain:
    def __init__(self):
        self.provider = None
        
        # PRIORITY 1: GOOGLE GEMINI (Free Tier)
        if os.getenv("GOOGLE_API_KEY"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                
                # Attempt to find best model automatically
                available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                preferred = ['models/gemini-2.5-flash']
                model_name = next((m for m in preferred if m in available), available[0])
                
                self.client = genai.GenerativeModel(model_name)
                self.provider = "google"
                print(f"{Fore.CYAN}🧠 Brain: Connected to {model_name} (Free Tier)")
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Failed to initialize Google Gemini: {e}")
            
        # PRIORITY 2: OPENROUTER (Paid Backup)
        if not self.provider and os.getenv("OPENROUTER_API_KEY"):
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
                self.provider = "openrouter"
                print(f"{Fore.CYAN}🧠 Brain: Connected to OpenRouter (Claude 3.5)")
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Failed to initialize OpenRouter: {e}")
            
        if not self.provider:
            print(f"{Fore.RED}❌ Error: No API Keys found in .env!")
            print("Please get a FREE key from: https://aistudio.google.com/app/apikey")
            sys.exit(1)

        # Chat history storage
        self.history = []

    def generate(self, system_prompt, user_input):
        """
        Unified generation function that handles history internally
        """
        # --- GEMINI LOGIC ---
        if self.provider == "google":
            # Reconstruction of context for Gemini
            full_context = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nCONVERSATION HISTORY:\n"
            
            for msg in self.history:
                role = "User" if msg['role'] == 'user' else "Model"
                full_context += f"{role}: {msg['content']}\n"
            
            # Add current user input
            full_context += f"User: {user_input}\nModel (Output ONLY Code):"
            
            try:
                # Use a fresh chat session for each turn to avoid state bloat
                # but pass the history in the prompt for simplicity and consistency
                response = self.client.generate_content(full_context)
                return self._clean_output(response.text)
            except Exception as e:
                return f"API Error (Gemini): {str(e)}"

        # --- OPENROUTER LOGIC ---
        elif self.provider == "openrouter":
            # Standard OpenAI format
            messages = [{"role": "system", "content": system_prompt}] + self.history + [{"role": "user", "content": user_input}]
            
            try:
                response = self.client.chat.completions.create(
                    model="anthropic/claude-3.5-sonnet",
                    messages=messages
                )
                return self._clean_output(response.choices[0].message.content)
            except Exception as e:
                return f"API Error (OpenRouter): {str(e)}"

    def _clean_output(self, text):
        """Removes markdown backticks if the AI adds them"""
        # Remove ```python and ``` blocks
        text = text.replace("```python", "").replace("```sonic", "").replace("```", "").strip()
        # Some models add "Here is the code:" - we remove lines starting with text if they aren't comments/imports
        lines = text.split('\n')
        clean_lines = [l for l in lines if not l.lower().strip().startswith("here is")]
        return "\n".join(clean_lines)

    def add_to_history(self, role, content):
        self.history.append({"role": role, "content": content})

# --- CORE PIPELINE ---

def run_pipeline(code):
    # --- THE FIX: Force a trailing newline ---
    # This prevents the "Unexpected token $END" error
    if not code.endswith("\n"):
        code += "\n"
        
    filename = "temp_gen.sonic"
    with open(filename, "w") as f:
        f.write(code)

    print(f"{Fore.YELLOW}⚙️  Compiling...")
    
    # Run the 'sonic' CLI
    try:
        process = subprocess.run(
            ["sonic", filename], 
            capture_output=True, 
            text=True
        )
    except FileNotFoundError:
        process = subprocess.run(
            [sys.executable, "main.py", filename], 
            capture_output=True, 
            text=True
        )

    # CASE 1: CRITICAL FAILURE (Syntax Error / Python Crash)
    if process.returncode != 0:
        error_log = process.stderr.strip()
        # Fallback: sometimes errors go to stdout
        if not error_log: error_log = process.stdout.strip()
        
        return False, f"COMPILER CRASH:\n{error_log}"
    
    # CASE 2: LOGIC FAILURE (Vibe Check)
    output_log = process.stdout
    if "VibeCheckError" in output_log:
        print(f"{Fore.WHITE}{output_log}") # DEBUG: Show all prints
        # Extract just the error line
        for line in output_log.split("\n"):
            if "VibeCheckError" in line:
                return False, f"VIBE FAIL: {line}"

    return True, "Success"

def main():
    print(f"{Fore.GREEN}🎹 White-Box Conductor (v2.1 Universal)")
    print(f"{Fore.CYAN}=========================================")

    # 1. Initialize Brain
    brain = AIBrain()
    
    # 2. Load System Prompt
    try:
        with open("system_prompt.txt", "r") as f:
            SYSTEM_PROMPT = f.read()
    except FileNotFoundError:
        print(f"{Fore.RED}❌ Error: system_prompt.txt missing!")
        sys.exit(1)

    # 3. Interactive Loop
    while True:
        user_input = input(f"\n{Fore.GREEN}You: ")
        if user_input.lower() in ['exit', 'quit', 'stop']: break
        
        print(f"{Fore.BLUE}🤖 Composing...")
        
        # Retry loop for syntax errors (Auto-Fix)
        current_prompt = user_input
        for attempt in range(3):
            ai_code = brain.generate(SYSTEM_PROMPT, current_prompt)
            
            if "API Error" in ai_code:
                print(f"{Fore.RED}{ai_code}")
                break

            success, msg = run_pipeline(ai_code)
            
            if success:
                print(f"{Fore.MAGENTA}✅ Code compiled successfully.")
                print(f"{Fore.WHITE}🔊 Playing audio...")
                
                # Save successful turn to history
                brain.add_to_history("user", user_input)
                brain.add_to_history("assistant", ai_code)
                break
            else:
                print(f"{Fore.RED}⚠️ Attempt {attempt+1} Failed: {msg}")
                # Feed the error back for the next retry
                current_prompt = f"The previous code failed with this error: {msg}. Please fix the syntax.\nCode was:\n{ai_code}"
                # Add failure to history to let AI learn from mistake
                brain.add_to_history("assistant", ai_code)
                brain.add_to_history("user", f"Error encountered: {msg}")

if __name__ == "__main__":
    main()
