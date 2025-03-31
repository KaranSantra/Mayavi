import socket
import json
import torch
import torchaudio
import os
from rich.console import Console
from generator import load_csm_1b, Segment


class CSMModule:
    def __init__(self, host="localhost", port=5002):
        self.console = Console()
        self.console.print("[bold blue]Initializing CSM Module...[/bold blue]")

        # Auto-detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.console.print(
            f"[blue]Device detection: Using {self.device.upper()}[/blue]"
        )
        self.running = False

        # Initialize CSM
        try:
            self.console.print("[blue]Loading CSM 1B model...[/blue]")
            self.generator = load_csm_1b(device=self.device)
            self.console.print(
                f"[green]✓ Successfully loaded CSM model on {self.device}[/green]"
            )

            # Load context audio
            self.console.print(
                "[blue]Loading context audio file 'maya-speaking-15.wav'...[/blue]"
            )
            self.context_audio = self.load_audio("maya-speaking-15.wav")
            self.context_segment = Segment(
                text="",  # Empty text as we just need the voice
                speaker=1,
                audio=self.context_audio,
            )
            self.console.print("[green]✓ Successfully loaded context audio[/green]")
        except Exception as e:
            self.console.print(
                f"[red bold]❌ Critical Error initializing CSM:[/red bold]"
            )
            self.console.print(f"[red]{str(e)}[/red]")
            raise

        # Initialize socket server
        self.console.print(
            f"\n[blue]Setting up socket server on {host}:{port}...[/blue]"
        )
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        self.console.print(
            f"[green]✓ Socket server successfully initialized and listening on {host}:{port}[/green]"
        )

    def load_audio(self, audio_path):
        """Load and preprocess audio for CSM"""
        self.console.print(f"[blue]Loading audio file: {audio_path}[/blue]")
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        self.console.print(f"[blue]Original sample rate: {sample_rate}Hz[/blue]")

        self.console.print(
            f"[blue]Resampling audio to {self.generator.sample_rate}Hz...[/blue]"
        )
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0),
            orig_freq=sample_rate,
            new_freq=self.generator.sample_rate,
        )
        self.console.print("[green]✓ Audio preprocessing complete[/green]")
        return audio_tensor

    def generate_voice(self, text):
        """Generate voice for the given text using CSM"""
        self.console.print("\n[blue]Starting voice generation process...[/blue]")
        self.console.print(f"[blue]Input text:[/blue] {text}")
        try:
            self.console.print("[blue]Generating audio with CSM model...[/blue]")
            audio = self.generator.generate(
                text=text,
                speaker=1,
                context=[self.context_segment],
                max_audio_length_ms=30_000,
            )
            self.console.print("[green]✓ Voice generation successful[/green]")
            return audio
        except Exception as e:
            self.console.print("[red bold]❌ Voice generation failed:[/red bold]")
            self.console.print(f"[red]{str(e)}[/red]")
            return None

    def start(self):
        """Start the CSM service"""
        self.running = True
        self.console.print("\n[bold green]=== CSM Service Started ===[/bold green]")
        self.console.print("[blue]Waiting for incoming connections...[/blue]\n")

        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                self.console.print(
                    f"[green]✓ New client connection established[/green]"
                )
                self.console.print(f"[blue]Client address: {address}[/blue]")

                while self.running:
                    # Receive data from client
                    self.console.print("\n[blue]Waiting for client message...[/blue]")
                    data = client_socket.recv(4096).decode("utf-8")
                    if not data:
                        self.console.print(
                            "[yellow]Client disconnected (empty data received)[/yellow]"
                        )
                        break

                    try:
                        message = json.loads(data)
                        self.console.print(
                            f"[blue]Received message type: {message['type']}[/blue]"
                        )

                        if message["type"] == "generate_voice":
                            text = message["text"]
                            self.console.print(
                                "\n[cyan]====== Voice Generation Request ======[/cyan]"
                            )
                            self.console.print(
                                f"[cyan]Text to synthesize:[/cyan] {text}"
                            )

                            # Generate voice
                            self.console.print(
                                "\n[blue]Processing voice generation...[/blue]"
                            )
                            audio = self.generate_voice(text)

                            if audio is not None:
                                # Save the audio temporarily
                                temp_file = os.path.abspath("temp_response.wav")
                                self.console.print(
                                    f"[blue]Saving generated audio to: {temp_file}[/blue]"
                                )
                                torchaudio.save(
                                    temp_file,
                                    audio.unsqueeze(0).cpu(),
                                    self.generator.sample_rate,
                                )
                                self.console.print(
                                    "[green]✓ Audio file saved successfully[/green]"
                                )

                                response = {
                                    "type": "voice_generated",
                                    "status": "success",
                                    "audio_file": temp_file,
                                }
                            else:
                                self.console.print(
                                    "[red]❌ Voice generation failed[/red]"
                                )
                                response = {
                                    "type": "voice_generated",
                                    "status": "error",
                                    "message": "Failed to generate voice",
                                }

                            self.console.print(
                                "[blue]Sending response to client...[/blue]"
                            )
                            client_socket.send(
                                f"{json.dumps(response)}\n".encode("utf-8")
                            )
                            self.console.print(
                                "[green]✓ Response sent to client[/green]"
                            )

                    except json.JSONDecodeError:
                        self.console.print(
                            "[red]❌ Error: Invalid JSON format received[/red]"
                        )
                        continue
                    except Exception as e:
                        self.console.print("[red]❌ Error processing message:[/red]")
                        self.console.print(f"[red]{str(e)}[/red]")
                        continue

                client_socket.close()
                self.console.print(
                    f"\n[yellow]Client connection closed: {address}[/yellow]"
                )

            except Exception as e:
                if self.running:
                    self.console.print("[red]❌ Socket error:[/red]")
                    self.console.print(f"[red]{str(e)}[/red]")
                break

    def stop(self):
        """Stop the CSM service"""
        self.running = False
        self.server_socket.close()
        self.console.print("\n[yellow]=== CSM Service Shutdown ===[/yellow]")
        self.console.print("[yellow]All connections closed.[/yellow]")


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
