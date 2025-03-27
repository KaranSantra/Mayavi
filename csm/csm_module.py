import socket
import json
import torch
import torchaudio
import os
from rich.console import Console
from generator import load_csm_1b, Segment


class CSMModule:
    def __init__(self, host="localhost", port=5002, device="cpu"):
        self.console = Console()
        self.device = device
        self.running = False

        # Initialize CSM
        try:
            self.generator = load_csm_1b(device=device)
            self.console.print("[green]Loaded CSM model[/green]")

            # Load context audio
            self.context_audio = self.load_audio("maya-speaking-15.wav")
            self.context_segment = Segment(
                text="",  # Empty text as we just need the voice
                speaker=1,
                audio=self.context_audio,
            )
        except Exception as e:
            self.console.print(f"[red]Error initializing CSM: {e}[/red]")
            raise

        # Initialize socket server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        self.console.print(f"[green]CSM service listening on {host}:{port}[/green]")

    def load_audio(self, audio_path):
        """Load and preprocess audio for CSM"""
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0),
            orig_freq=sample_rate,
            new_freq=self.generator.sample_rate,
        )
        return audio_tensor

    def generate_voice(self, text):
        """Generate voice for the given text using CSM"""
        try:
            audio = self.generator.generate(
                text=text,
                speaker=1,
                context=[self.context_segment],
                max_audio_length_ms=30_000,
            )
            return audio
        except Exception as e:
            self.console.print(f"[red]Error generating voice: {e}[/red]")
            return None

    def start(self):
        """Start the CSM service"""
        self.running = True
        self.console.print("\n[bold green]Started CSM Service![/bold green]")
        self.console.print("Waiting for connections...\n")

        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                self.console.print(f"[green]Connected to client at {address}[/green]")

                while self.running:
                    # Receive data from client
                    data = client_socket.recv(4096).decode("utf-8")
                    if not data:
                        break

                    try:
                        message = json.loads(data)
                        if message["type"] == "generate_voice":
                            text = message["text"]
                            self.console.print(
                                f"[cyan]Generating voice for text: {text[:50]}...[/cyan]"
                            )

                            # Generate voice
                            audio = self.generate_voice(text)
                            if audio is not None:
                                # Save the audio temporarily with absolute path
                                temp_file = os.path.abspath("temp_response.wav")
                                torchaudio.save(
                                    temp_file,
                                    audio.unsqueeze(0).cpu(),
                                    self.generator.sample_rate,
                                )

                                # Send success response with absolute audio file path
                                response = {
                                    "type": "voice_generated",
                                    "status": "success",
                                    "audio_file": temp_file,
                                }
                            else:
                                response = {
                                    "type": "voice_generated",
                                    "status": "error",
                                    "message": "Failed to generate voice",
                                }

                            client_socket.send(
                                f"{json.dumps(response)}\n".encode("utf-8")
                            )

                    except json.JSONDecodeError:
                        self.console.print("[red]Invalid JSON received[/red]")
                        continue
                    except Exception as e:
                        self.console.print(f"[red]Error processing message: {e}[/red]")
                        continue

                client_socket.close()
                self.console.print(f"[yellow]Client disconnected: {address}[/yellow]")

            except Exception as e:
                if self.running:
                    self.console.print(f"[red]Error: {e}[/red]")
                break

    def stop(self):
        """Stop the CSM service"""
        self.running = False
        self.server_socket.close()
        self.console.print("\n[yellow]CSM service stopped.[/yellow]")


def main():
    module = None
    try:
        module = CSMModule()
        module.start()
    except KeyboardInterrupt:
        module.console.print("\n[yellow]Stopping CSM service...[/yellow]")
    except Exception as e:
        module.console.print(f"\n[red]Error: {e}[/red]")
    finally:
        if module:
            module.stop()


if __name__ == "__main__":
    main()
