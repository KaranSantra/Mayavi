import asyncio
import websockets
import json
import socket
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import print as rprint
from rich.table import Table
from config import (
    setup_logging,
    SERVER_INTERNAL_HOST,
    SERVER_PORT,
    CHUNK_SIZE,
    SAMPLE_RATE,
)


class AudioServer:
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
        self.logger = setup_logging("server")

        # Initialize sockets for ASR/LLM/CSM services
        self.asr_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.llm_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.csm_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to services
        self.connect_to_services(
            asr_host, asr_port, llm_host, llm_port, csm_host, csm_port
        )

        # WebSocket server
        self.clients = set()

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

    def connect_to_services(
        self, asr_host, asr_port, llm_host, llm_port, csm_host, csm_port
    ):
        """Connect to ASR, LLM, and CSM services"""
        try:
            self.console.print(f"[yellow]⏳ Connecting to services...[/yellow]")

            # Connect to ASR service
            self.asr_socket.connect((asr_host, asr_port))
            self.console.print(
                f"[green]✓ Connected to ASR service at {asr_host}:{asr_port}[/green]"
            )

            # Connect to LLM service
            self.llm_socket.connect((llm_host, llm_port))
            self.console.print(
                f"[green]✓ Connected to LLM service at {llm_host}:{llm_port}[/green]"
            )

            # Connect to CSM service
            self.csm_socket.connect((csm_host, csm_port))
            self.console.print(
                f"[green]✓ Connected to CSM service at {csm_host}:{csm_port}[/green]"
            )

        except ConnectionRefusedError as e:
            self.console.print(f"[red]✗ Failed to connect to services: {str(e)}[/red]")
            raise

    async def handle_client(self, websocket, path):
        """Handle individual client connections"""
        client_id = id(websocket)
        self.clients.add(websocket)

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.console.print(
            f"[{timestamp}] [blue]➤ New client connected (ID: {client_id})[/blue]"
        )

        try:
            # Send start command to ASR service
            start_command = {"type": "start"}
            self.asr_socket.send(json.dumps(start_command).encode("utf-8"))

            while True:
                message = await websocket.recv()
                data = json.loads(message)
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                if data["type"] == "audio_data":
                    audio_data = bytes.fromhex(data["data"])

                    # Print audio data details
                    self.console.print(
                        f"[{timestamp}] [cyan]📥 Received audio chunk:[/cyan]\n"
                        f"  Size: {len(audio_data)} bytes\n"
                        f"  First 8 bytes: {audio_data[:8].hex()}"
                    )

                    # Send to ASR service
                    self.asr_socket.send(audio_data)
                    self.console.print(
                        f"[{timestamp}] [cyan]➤ Sent to ASR service[/cyan]"
                    )

                    # Get ASR response
                    asr_response = self.asr_socket.recv(4096).decode("utf-8")
                    if asr_response:
                        asr_data = json.loads(asr_response)
                        if asr_data["type"] == "transcription":
                            transcribed_text = asr_data["text"].strip()
                            if transcribed_text:
                                self.console.print(
                                    f"[{timestamp}] [green]🎯 Transcription:[/green] {transcribed_text}"
                                )

                                # Send to LLM
                                self.console.print(
                                    f"[{timestamp}] [cyan]➤ Sending to LLM...[/cyan]"
                                )
                                self.send_to_llm(transcribed_text)
                                llm_response = self.receive_from_llm()

                                if llm_response:
                                    self.console.print(
                                        f"[{timestamp}] [green]💡 LLM Response:[/green] {llm_response}"
                                    )

                                    # Generate voice response
                                    self.console.print(
                                        f"[{timestamp}] [cyan]➤ Requesting voice generation...[/cyan]"
                                    )
                                    self.send_to_csm(llm_response)
                                    audio_file = self.receive_from_csm()

                                    if audio_file:
                                        with open(audio_file, "rb") as f:
                                            audio_data = f.read()
                                            self.console.print(
                                                f"[{timestamp}] [cyan]📤 Sending audio response:[/cyan]\n"
                                                f"  File: {audio_file}\n"
                                                f"  Size: {len(audio_data)} bytes\n"
                                                f"  First 8 bytes: {audio_data[:8].hex()}"
                                            )
                                            await websocket.send(
                                                json.dumps(
                                                    {
                                                        "type": "audio_response",
                                                        "data": audio_data.hex(),
                                                    }
                                                )
                                            )

        except websockets.exceptions.ConnectionClosed:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.console.print(
                f"[{timestamp}] [red]✗ Client {client_id} disconnected[/red]"
            )
        except Exception as e:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.console.print(
                f"[{timestamp}] [red]✗ Error handling client {client_id}: {str(e)}[/red]"
            )
        finally:
            self.clients.remove(websocket)

    def send_to_llm(self, text):
        """Send message to LLM service"""
        message_data = {"type": "transcription", "text": text}
        try:
            self.llm_socket.sendall(f"{json.dumps(message_data)}\n".encode("utf-8"))
        except Exception as e:
            self.logger.error(f"Error sending to LLM: {str(e)}")
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
            self.logger.error(f"Error receiving from LLM: {str(e)}")
        return None

    def send_to_csm(self, text):
        """Send message to CSM service"""
        message_data = {"type": "generate_voice", "text": text}
        try:
            self.csm_socket.sendall(f"{json.dumps(message_data)}\n".encode("utf-8"))
        except Exception as e:
            self.logger.error(f"Error sending to CSM: {str(e)}")
            raise

    def receive_from_csm(self):
        """Receive response from CSM service"""
        try:
            data = self.csm_socket.recv(4096)
            if not data:
                return None

            messages = data.decode("utf-8").strip().split("\n")
            response_data = json.loads(messages[-1])

            if (
                response_data.get("type") == "voice_generated"
                and response_data.get("status") == "success"
            ):
                return response_data.get("audio_file")
        except Exception as e:
            self.logger.error(f"Error receiving from CSM: {str(e)}")
        return None

    async def run(self):
        """Main server loop"""
        try:
            server = await websockets.serve(
                self.handle_client, SERVER_INTERNAL_HOST, SERVER_PORT
            )

            self.console.print(
                Panel(
                    f"Server running on ws://{SERVER_INTERNAL_HOST}:{SERVER_PORT}",
                    style="green bold",
                    title="Server Status",
                )
            )

            # Keep server running
            await asyncio.Future()  # run forever

        except KeyboardInterrupt:
            self.console.print("[yellow]Server shutting down...[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Error in main loop: {str(e)}[/red]")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        # Close service connections
        self.asr_socket.close()
        self.llm_socket.close()
        self.csm_socket.close()

        # Close all client connections
        for client in self.clients:
            asyncio.create_task(client.close())


def main():
    server = AudioServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
