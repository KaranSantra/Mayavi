import asyncio
import websockets
import pyaudio
import sounddevice as sd
import numpy as np
import json
import queue
import threading
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from config import (
    setup_logging,
    SERVER_HOST,
    SERVER_PORT,
    CHUNK_SIZE,
    SAMPLE_RATE,
    CHANNELS,
    FORMAT,
)


class AudioClient:
    def __init__(self, server_host=SERVER_HOST, server_port=SERVER_PORT):
        self.console = Console()
        self.logger = setup_logging("client")
        self.server_host = server_host
        self.server_port = server_port

        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.audio.get_format_from_width(2),
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        # Audio playback queue
        self.audio_queue = queue.Queue()
        self.is_playing = False

        # WebSocket connection
        self.ws = None
        self.connected = False

        # UI components
        self.layout = Layout()
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        self.status_text = Text("", style="bold")
        self.transcription_text = Text("", style="green")
        self.response_text = Text("", style="blue")

    async def connect(self):
        """Establish WebSocket connection to server"""
        try:
            self.ws = await websockets.connect(
                f"ws://{self.server_host}:{self.server_port}"
            )
            self.connected = True
            self.status_text.append("Connected to server", style="green")
            self.logger.info("Connected to server")
        except Exception as e:
            self.status_text.append(f"Connection failed: {str(e)}", style="red")
            self.logger.error(f"Connection failed: {str(e)}")
            raise

    def audio_capture_thread(self):
        """Thread for capturing audio from microphone"""
        while self.connected:
            try:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                asyncio.run(self.send_audio(data))
            except Exception as e:
                self.logger.error(f"Error capturing audio: {str(e)}")
                break

    async def send_audio(self, audio_data):
        """Send audio data to server"""
        if self.connected:
            try:
                await self.ws.send(
                    json.dumps(
                        {
                            "type": "audio_data",
                            "data": audio_data.hex(),  # Convert bytes to hex string for JSON
                        }
                    )
                )
            except Exception as e:
                self.logger.error(f"Error sending audio: {str(e)}")
                self.connected = False

    def audio_playback_thread(self):
        """Thread for playing received audio"""
        while self.connected:
            try:
                audio_data = self.audio_queue.get()
                if audio_data is None:  # Poison pill to stop the thread
                    break
                sd.play(audio_data, SAMPLE_RATE)
                sd.wait()
            except Exception as e:
                self.logger.error(f"Error playing audio: {str(e)}")
                break

    async def receive_messages(self):
        """Handle incoming messages from server"""
        while self.connected:
            try:
                message = await self.ws.recv()
                data = json.loads(message)

                if data["type"] == "transcription":
                    self.transcription_text.append(f"\nYou: {data['text']}")
                    self.logger.info(f"Transcription: {data['text']}")

                elif data["type"] == "llm_response":
                    self.response_text.append(f"\nAssistant: {data['text']}")
                    self.logger.info(f"LLM Response: {data['text']}")

                elif data["type"] == "audio_response":
                    # Convert hex string back to bytes
                    audio_data = bytes.fromhex(data["data"])
                    self.audio_queue.put(np.frombuffer(audio_data, dtype=np.int16))
                    self.logger.debug("Received audio response")

            except Exception as e:
                self.logger.error(f"Error receiving message: {str(e)}")
                self.connected = False
                break

    def update_display(self):
        """Update the rich console display"""
        self.layout["header"].update(Panel(self.status_text, title="Status"))
        self.layout["main"].update(
            Panel(
                f"{self.transcription_text}\n\n{self.response_text}",
                title="Conversation",
                border_style="white",
            )
        )
        self.layout["footer"].update(Panel("Press Ctrl+C to exit", style="yellow"))

    async def run(self):
        """Main client loop"""
        try:
            await self.connect()

            # Start audio capture and playback threads
            capture_thread = threading.Thread(target=self.audio_capture_thread)
            playback_thread = threading.Thread(target=self.audio_playback_thread)

            capture_thread.start()
            playback_thread.start()

            # Start message receiver
            receiver_task = asyncio.create_task(self.receive_messages())

            # Main display loop
            with Live(self.layout, refresh_per_second=4) as live:
                while self.connected:
                    self.update_display()
                    await asyncio.sleep(0.25)

            # Cleanup
            self.connected = False
            self.audio_queue.put(None)  # Stop playback thread
            capture_thread.join()
            playback_thread.join()
            await receiver_task

        except KeyboardInterrupt:
            self.logger.info("Client shutting down...")
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        if self.ws:
            asyncio.run(self.ws.close())
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


def main():
    client = AudioClient()
    asyncio.run(client.run())


if __name__ == "__main__":
    main()
