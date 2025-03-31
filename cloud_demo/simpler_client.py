import asyncio
import websockets
import pyaudio
import json
import os
import time
import threading
from datetime import datetime
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
import wave
from pathlib import Path
from config import SERVER_EXTERNAL_HOST, SERVER_PORT


class SimpleAudioClient:
    def __init__(self, server_host=SERVER_EXTERNAL_HOST, server_port=SERVER_PORT):
        # Audio configuration
        self.CHUNK = 1024 * 3
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000

        # Setup components
        self.console = Console()
        self.setup_audio()
        self.setup_queues()
        self.setup_temp_directory()

        # Connection info
        self.server_uri = f"ws://{server_host}:{server_port}"
        self.ws = None
        self.connected = False

        # State tracking
        self.running = False
        self.is_playing = False

        self.console.print("[yellow]Initializing audio client...[/yellow]")

    def setup_audio(self):
        """Initialize PyAudio for recording and playback"""
        try:
            self.console.print("[yellow]Setting up audio device...[/yellow]")
            self.audio = pyaudio.PyAudio()
            self.console.print("[blue]Created PyAudio instance[/blue]")

            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
            )
            self.console.print(f"[blue]Opened audio stream with parameters:[/blue]")
            self.console.print(f"  • Format: {self.FORMAT}")
            self.console.print(f"  • Channels: {self.CHANNELS}")
            self.console.print(f"  • Rate: {self.RATE} Hz")
            self.console.print(f"  • Chunk Size: {self.CHUNK}")
            self.console.print("[green]✓ Audio device successfully initialized[/green]")
        except Exception as e:
            self.console.print(f"[red]× Audio setup failed: {e}[/red]")
            raise

    def setup_queues(self):
        """Initialize audio queues"""
        self.console.print("[yellow]Setting up audio queues...[/yellow]")
        self.capture_queue = asyncio.Queue(maxsize=32)
        self.console.print("[green]✓ Audio queue initialized (max size: 32)[/green]")

    def setup_temp_directory(self):
        """Setup directory for temporary audio files"""
        self.console.print("[yellow]Setting up temporary directory...[/yellow]")
        self.temp_dir = Path("temp_audio")
        self.temp_dir.mkdir(exist_ok=True)
        self.console.print(f"[blue]Created temp directory: {self.temp_dir}[/blue]")

        # Cleanup old files
        cleaned = 0
        for file in self.temp_dir.glob("*.wav"):
            if time.time() - file.stat().st_mtime > 86400:  # 24 hours
                file.unlink()
                cleaned += 1
        if cleaned > 0:
            self.console.print(f"[blue]Cleaned up {cleaned} old audio files[/blue]")
        self.console.print("[green]✓ Temporary directory setup complete[/green]")

    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.console.print(
                f"[yellow]Connecting to server at {self.server_uri}...[/yellow]"
            )
            self.ws = await websockets.connect(self.server_uri)
            self.connected = True
            self.console.print("[green]✓ Connected to server successfully[/green]")
            return True
        except Exception as e:
            self.console.print(f"[red]× Connection failed: {e}[/red]")
            return False

    def audio_callback(self):
        """Capture audio data and put in queue"""
        self.console.print("[yellow]Starting audio capture...[/yellow]")
        chunks_processed = 0
        while self.running:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                asyncio.run_coroutine_threadsafe(
                    self.capture_queue.put(data), self.loop
                )
                chunks_processed += 1
                if chunks_processed % 100 == 0:  # Log every 100 chunks
                    self.console.print(
                        f"[blue]Processed {chunks_processed} audio chunks[/blue]"
                    )
            except Exception as e:
                self.console.print(f"[red]× Audio capture error: {e}[/red]")
                if not self.running:
                    break
        self.console.print("[yellow]Audio capture stopped[/yellow]")

    async def process_audio_queue(self):
        """Process and send captured audio"""
        self.console.print("[yellow]Starting audio queue processing...[/yellow]")
        packets_sent = 0
        while self.running:
            try:
                data = await self.capture_queue.get()
                if self.connected:
                    await self.ws.send(
                        json.dumps({"type": "audio_data", "data": data.hex()})
                    )
                    packets_sent += 1
                    if packets_sent % 100 == 0:  # Log every 100 packets
                        self.console.print(
                            f"[blue]Sent {packets_sent} audio packets to server[/blue]"
                        )
            except Exception as e:
                self.console.print(f"[red]× Error sending audio: {e}[/red]")
                if not self.connected:
                    break

    async def handle_server_messages(self):
        """Process incoming server messages"""
        self.console.print("[yellow]Starting server message handler...[/yellow]")
        while self.connected:
            try:
                message = await self.ws.recv()
                data = json.loads(message)

                if data["type"] == "transcription":
                    self.console.print(f"[cyan]You: {data['text']}[/cyan]")
                elif data["type"] == "llm_response":
                    self.console.print(f"[green]Assistant: {data['text']}[/green]")
                elif data["type"] == "audio_response":
                    await self.handle_audio_response(data["data"])

            except Exception as e:
                self.console.print(f"[red]× Error receiving message: {e}[/red]")
                if not self.connected:
                    break

    async def handle_audio_response(self, audio_data):
        """Save and play received audio"""
        try:
            self.console.print("[yellow]Processing received audio...[/yellow]")

            # Generate unique filename
            filename = self.temp_dir / f"response_{int(time.time())}.wav"
            self.console.print(f"[blue]Saving audio to: {filename}[/blue]")

            # Save audio data
            audio_bytes = bytes.fromhex(audio_data)
            with wave.open(str(filename), "wb") as wave_file:
                wave_file.setnchannels(self.CHANNELS)
                wave_file.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wave_file.setframerate(self.RATE)
                wave_file.writeframes(audio_bytes)
            self.console.print("[green]✓ Audio file saved successfully[/green]")

            # Play audio
            self.console.print("[yellow]Starting audio playback...[/yellow]")
            await self.play_audio(filename)

        except Exception as e:
            self.console.print(f"[red]× Error handling audio response: {e}[/red]")

    async def play_audio(self, filename):
        """Play audio from file"""
        try:
            self.is_playing = True
            self.console.print(f"[yellow]Playing audio: {filename}[/yellow]")

            with wave.open(str(filename), "rb") as wave_file:
                duration = wave_file.getnframes() / wave_file.getframerate()
                self.console.print(
                    f"[blue]Audio duration: {duration:.2f} seconds[/blue]"
                )

                stream = self.audio.open(
                    format=self.audio.get_format_from_width(wave_file.getsampwidth()),
                    channels=wave_file.getnchannels(),
                    rate=wave_file.getframerate(),
                    output=True,
                )

                data = wave_file.readframes(self.CHUNK)
                frames_played = 0
                total_frames = wave_file.getnframes()

                while data and self.is_playing:
                    stream.write(data)
                    data = wave_file.readframes(self.CHUNK)
                    frames_played += len(data)
                    if (
                        frames_played % (total_frames // 4) == 0
                    ):  # Log progress at 25%, 50%, 75%
                        progress = (frames_played / total_frames) * 100
                        self.console.print(
                            f"[blue]Playback progress: {progress:.0f}%[/blue]"
                        )

                stream.stop_stream()
                stream.close()
                self.console.print("[green]✓ Audio playback completed[/green]")

        except Exception as e:
            self.console.print(f"[red]× Error playing audio: {e}[/red]")
        finally:
            self.is_playing = False

    async def run(self):
        """Main client loop"""
        self.running = True
        self.loop = asyncio.get_event_loop()

        self.console.print("[yellow]Starting audio client...[/yellow]")

        if not await self.connect():
            return

        self.console.print("[yellow]Initializing audio capture thread...[/yellow]")
        capture_thread = threading.Thread(target=self.audio_callback)
        capture_thread.start()

        try:
            self.console.print("[green]✓ Client fully initialized and running[/green]")
            await asyncio.gather(
                self.process_audio_queue(), self.handle_server_messages()
            )
        except KeyboardInterrupt:
            self.console.print("[yellow]Received shutdown signal...[/yellow]")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.console.print("[yellow]Starting cleanup...[/yellow]")
        self.running = False
        self.connected = False

        if hasattr(self, "stream"):
            self.stream.stop_stream()
            self.stream.close()
            self.console.print("[blue]Closed audio stream[/blue]")

        if hasattr(self, "audio"):
            self.audio.terminate()
            self.console.print("[blue]Terminated PyAudio[/blue]")

        if self.ws:
            asyncio.create_task(self.ws.close())
            self.console.print("[blue]Closed WebSocket connection[/blue]")

        self.console.print("[green]✓ Cleanup completed[/green]")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple Audio Client")
    parser.add_argument("--host", default=SERVER_EXTERNAL_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server port")

    args = parser.parse_args()

    client = SimpleAudioClient(args.host, args.port)
    asyncio.run(client.run())


if __name__ == "__main__":
    main()
