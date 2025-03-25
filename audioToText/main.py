# audioToText/main.py
import os
import sys
import time
import subprocess
import socket
import json
import threading
import signal
import argparse
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner


class VoiceConversationSystem:
    def __init__(self, asr_port=5000, llm_port=5001):
        self.asr_port = asr_port
        self.llm_port = llm_port
        self.asr_socket = None
        self.llm_socket = None
        self.asr_process = None
        self.llm_process = None
        self.running = False

        # Create Rich console for pretty output
        self.console = Console()

        # Project paths
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.asr_path = os.path.join(self.project_root, "asr")
        self.llm_path = os.path.join(self.project_root, "llm")

        # Determine Python virtual environment paths
        self.asr_venv = os.path.join(self.asr_path, ".venv")
        self.llm_venv = os.path.join(self.llm_path, ".venv")

        # Adjust for Windows/Unix
        if sys.platform == "win32":
            self.asr_python = os.path.join(self.asr_venv, "Scripts", "python.exe")
            self.llm_python = os.path.join(self.llm_venv, "Scripts", "python.exe")
        else:
            self.asr_python = os.path.join(self.asr_venv, "bin", "python")
            self.llm_python = os.path.join(self.llm_venv, "bin", "python")

    def start_asr_module(self):
        """Start the ASR module in its virtual environment"""
        try:
            asr_module_path = os.path.join(self.asr_path, "asr_module.py")
            self.console.print("[yellow]Starting ASR Module...[/yellow]")

            self.asr_process = subprocess.Popen(
                [self.asr_python, asr_module_path, "--port", str(self.asr_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Start a thread to monitor and print ASR module output
            def monitor_asr_output():
                for line in self.asr_process.stdout:
                    self.console.print(f"[dim cyan][ASR][/dim cyan] {line.strip()}")

            asr_thread = threading.Thread(target=monitor_asr_output)
            asr_thread.daemon = True
            asr_thread.start()

            # Connect to ASR module
            self.console.print("[yellow]Connecting to ASR Module...[/yellow]")
            for i in range(10):  # try for 10 seconds
                try:
                    self.asr_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.asr_socket.connect(("127.0.0.1", self.asr_port))
                    self.console.print(
                        "[bold green]✓[/bold green] Connected to ASR Module!"
                    )
                    return True
                except (ConnectionRefusedError, socket.error):
                    time.sleep(1)

            self.console.print("[bold red]Failed to connect to ASR Module![/bold red]")
            return False

        except Exception as e:
            self.console.print(f"[bold red]Error starting ASR Module: {e}[/bold red]")
            return False

    def start_llm_module(self):
        """Start the LLM module in its virtual environment"""
        try:
            llm_module_path = os.path.join(self.llm_path, "llm_module.py")
            self.console.print("[yellow]Starting LLM Module...[/yellow]")

            self.llm_process = subprocess.Popen(
                [self.llm_python, llm_module_path, "--port", str(self.llm_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Start a thread to monitor and print LLM module output
            def monitor_llm_output():
                for line in self.llm_process.stdout:
                    self.console.print(
                        f"[dim magenta][LLM][/dim magenta] {line.strip()}"
                    )

            llm_thread = threading.Thread(target=monitor_llm_output)
            llm_thread.daemon = True
            llm_thread.start()

            # Connect to LLM module
            self.console.print("[yellow]Connecting to LLM Module...[/yellow]")
            for i in range(30):  # try for 30 seconds (LLM loading takes longer)
                try:
                    self.llm_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.llm_socket.connect(("127.0.0.1", self.llm_port))
                    self.console.print(
                        "[bold green]✓[/bold green] Connected to LLM Module!"
                    )
                    return True
                except (ConnectionRefusedError, socket.error):
                    time.sleep(1)

            self.console.print("[bold red]Failed to connect to LLM Module![/bold red]")
            return False

        except Exception as e:
            self.console.print(f"[bold red]Error starting LLM Module: {e}[/bold red]")
            return False

    def forward_transcription_to_llm(self, text):
        """Forward transcription from ASR to LLM module"""
        try:
            message = json.dumps({"type": "transcription", "text": text})
            self.llm_socket.sendall(f"{message}\n".encode("utf-8"))
        except Exception as e:
            self.console.print(
                f"[bold red]Error forwarding transcription: {e}[/bold red]"
            )

    def handle_messages(self):
        """Handle messages from ASR and LLM modules"""
        # Setup socket connections to read from both modules
        asr_thread = threading.Thread(target=self.read_from_asr)
        asr_thread.daemon = True
        asr_thread.start()

        llm_thread = threading.Thread(target=self.read_from_llm)
        llm_thread.daemon = True
        llm_thread.start()

    def read_from_asr(self):
        """Read and process messages from ASR module"""
        buffer = ""
        while self.running:
            try:
                data = self.asr_socket.recv(4096)
                if not data:
                    break

                buffer += data.decode("utf-8")
                messages = buffer.split("\n")
                buffer = messages.pop()  # Keep incomplete message

                for msg in messages:
                    try:
                        message = json.loads(msg)
                        if message.get("type") == "transcription":
                            text = message.get("text", "").strip()
                            if text:
                                self.console.print(
                                    Panel(text, title="You", border_style="green")
                                )
                                self.forward_transcription_to_llm(text)
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                if self.running:
                    self.console.print(
                        f"[bold red]Error reading from ASR: {e}[/bold red]"
                    )
                    time.sleep(1)

    def read_from_llm(self):
        """Read and process messages from LLM module"""
        buffer = ""
        while self.running:
            try:
                data = self.llm_socket.recv(4096)
                if not data:
                    break

                buffer += data.decode("utf-8")
                messages = buffer.split("\n")
                buffer = messages.pop()  # Keep incomplete message

                for msg in messages:
                    try:
                        message = json.loads(msg)
                        if message.get("type") == "llm_response":
                            text = message.get("text", "").strip()
                            if text:
                                self.console.print(
                                    Panel(
                                        Markdown(text),
                                        title="Assistant",
                                        border_style="blue",
                                    )
                                )
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                if self.running:
                    self.console.print(
                        f"[bold red]Error reading from LLM: {e}[/bold red]"
                    )
                    time.sleep(1)

    def start(self):
        """Start the voice conversation system"""
        self.console.print(
            "[bold blue]====== Voice Conversation System ======[/bold blue]"
        )

        # Make sure required files exist
        asr_module_path = os.path.join(self.asr_path, "asr_module.py")
        llm_module_path = os.path.join(self.llm_path, "llm_module.py")

        for path in [asr_module_path, llm_module_path]:
            if not os.path.exists(path):
                self.console.print(
                    f"[bold red]Error: Required file not found at {path}[/bold red]"
                )
                return False

        # Start modules
        if not self.start_asr_module():
            self.stop()
            return False

        if not self.start_llm_module():
            self.stop()
            return False

        self.running = True
        self.handle_messages()

        self.console.print("\n[bold green]System started successfully![/bold green]")
        self.console.print(
            "[cyan]Speak into your microphone to start a conversation...[/cyan]"
        )
        self.console.print("[dim]Press Ctrl+C to exit[/dim]\n")

        return True

    def stop(self):
        """Stop the voice conversation system"""
        self.running = False

        # Close socket connections
        if self.asr_socket:
            self.asr_socket.close()

        if self.llm_socket:
            self.llm_socket.close()

        # Terminate processes
        for process in [self.asr_process, self.llm_process]:
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except (subprocess.TimeoutExpired, Exception):
                    if sys.platform == "win32":
                        subprocess.call(
                            ["taskkill", "/F", "/T", "/PID", str(process.pid)]
                        )
                    else:
                        os.kill(process.pid, signal.SIGKILL)

        self.console.print("[yellow]System stopped[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Voice Conversation System")
    parser.add_argument(
        "--asr-port", type=int, default=5000, help="Port for ASR module"
    )
    parser.add_argument(
        "--llm-port", type=int, default=5001, help="Port for LLM module"
    )
    args = parser.parse_args()

    system = VoiceConversationSystem(asr_port=args.asr_port, llm_port=args.llm_port)

    def signal_handler(sig, frame):
        print("\nShutting down...")
        system.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        if system.start():
            # Keep main thread alive
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        system.stop()


if __name__ == "__main__":
    main()
