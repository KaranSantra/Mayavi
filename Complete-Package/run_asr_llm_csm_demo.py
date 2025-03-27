import sys
import os
import socket
import json
import time
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ASRLLMCSMDemo:
    def __init__(
        self,
        asr_host="localhost",
        asr_port=5000,
        llm_host="localhost",
        llm_port=5001,
        csm_host="localhost",
        csm_port=5002,
    ):
        self.console = Console()
        self.running = False

        # Initialize sockets
        self.asr_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.llm_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.csm_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to ASR service
        try:
            self.asr_socket.connect((asr_host, asr_port))
            self.console.print("[green]Connected to ASR service[/green]")
        except ConnectionRefusedError:
            self.console.print(
                "[red]Could not connect to ASR service. Make sure it's running.[/red]"
            )
            raise

        # Connect to LLM service
        try:
            self.llm_socket.connect((llm_host, llm_port))
            self.console.print("[green]Connected to LLM service[/green]")
        except ConnectionRefusedError:
            self.console.print(
                "[red]Could not connect to LLM service. Make sure it's running.[/red]"
            )
            self.asr_socket.close()
            raise

        # Connect to CSM service
        try:
            self.csm_socket.connect((csm_host, csm_port))
            self.console.print("[green]Connected to CSM service[/green]")
        except ConnectionRefusedError:
            self.console.print(
                "[red]Could not connect to CSM service. Make sure it's running.[/red]"
            )
            self.asr_socket.close()
            self.llm_socket.close()
            raise

    def start(self):
        """Start the demo"""
        self.running = True

        # Send start command to ASR service
        start_command = {"type": "start"}
        self.asr_socket.send(json.dumps(start_command).encode("utf-8"))

        self.console.print("\n[bold green]Started ASR-LLM-CSM Demo![/bold green]")
        self.console.print("Speak into your microphone. Press Ctrl+C to stop.\n")

        # Main processing loop
        while self.running:
            try:
                # Read from ASR service
                asr_start_time = time.time()
                asr_data = self.asr_socket.recv(1024).decode("utf-8")
                if not asr_data:
                    break

                message = json.loads(asr_data)
                if message["type"] == "transcription":
                    asr_end_time = time.time()
                    transcribed_text = message["text"].strip()
                    if transcribed_text:
                        # Print transcription with timestamp
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        self.console.print(
                            Panel(
                                f"{transcribed_text}\n\n[dim]Response time: {asr_end_time - asr_start_time:.2f}s[/dim]",
                                title=f"[cyan]You ({timestamp})[/cyan]",
                                border_style="green",
                            )
                        )

                        # Send to LLM and get response
                        start_time = time.time()
                        self.send_to_llm(transcribed_text)
                        response = self.receive_from_llm()
                        end_time = time.time()

                        if response:
                            # Print LLM response with timing
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            self.console.print(
                                Panel(
                                    f"{response}\n\n[dim]Response time: {end_time - start_time:.2f}s[/dim]",
                                    title=f"[cyan]Assistant ({timestamp})[/cyan]",
                                    border_style="blue",
                                )
                            )

                            # Generate and play voice response
                            voice_start_time = time.time()
                            self.send_to_csm(response)
                            audio_file = self.receive_from_csm()
                            if audio_file:
                                os.system(f"afplay {audio_file}")  # For macOS
                                os.remove(audio_file)  # Clean up
                            voice_end_time = time.time()
                            self.console.print(
                                f"[dim]Voice generation time: {voice_end_time - voice_start_time:.2f}s[/dim]"
                            )

            except Exception as e:
                if self.running:
                    self.console.print(f"[red]Error: {e}[/red]")
                break

    def send_to_llm(self, text):
        """Send message to LLM service"""
        message_data = {"type": "transcription", "text": text}
        try:
            self.llm_socket.sendall(f"{json.dumps(message_data)}\n".encode("utf-8"))
        except Exception as e:
            self.console.print(f"[red]Error sending to LLM: {e}[/red]")
            raise

    def receive_from_llm(self):
        """Receive response from LLM service"""
        try:
            data = self.llm_socket.recv(4096)
            if not data:
                return None

            response_data = json.loads(data.decode("utf-8"))
            if response_data.get("type") == "llm_response":
                return response_data.get("text")
        except Exception as e:
            self.console.print(f"[red]Error receiving from LLM: {e}[/red]")
        return None

    def send_to_csm(self, text):
        """Send message to CSM service"""
        message_data = {"type": "generate_voice", "text": text}
        try:
            self.csm_socket.sendall(f"{json.dumps(message_data)}\n".encode("utf-8"))
        except Exception as e:
            self.console.print(f"[red]Error sending to CSM: {e}[/red]")
            raise

    def receive_from_csm(self):
        """Receive response from CSM service"""
        try:
            data = self.csm_socket.recv(4096)
            if not data:
                return None

            # Split the received data by newlines and process the last complete message
            messages = data.decode("utf-8").strip().split("\n")
            response_data = json.loads(messages[-1])  # Take the last message

            if (
                response_data.get("type") == "voice_generated"
                and response_data.get("status") == "success"
            ):
                return response_data.get("audio_file")
        except Exception as e:
            self.console.print(f"[red]Error receiving from CSM: {e}[/red]")
        return None

    def stop(self):
        """Stop the demo"""
        self.running = False

        # Send stop command to ASR service
        stop_command = {"type": "stop"}
        try:
            self.asr_socket.send(json.dumps(stop_command).encode("utf-8"))
        except:
            pass

        # Close connections
        self.asr_socket.close()
        self.llm_socket.close()
        self.csm_socket.close()
        self.console.print("\n[yellow]Demo ended.[/yellow]")


def main():
    demo = None
    try:
        demo = ASRLLMCSMDemo()
        demo.start()
    except KeyboardInterrupt:
        demo.console.print("\n[yellow]Stopping demo...[/yellow]")
    except Exception as e:
        demo.console.print(f"\n[red]Error: {e}[/red]")
    finally:
        if demo:
            demo.stop()


if __name__ == "__main__":
    main()
