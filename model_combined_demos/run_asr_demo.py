import socket
import json
from rich.console import Console


class ASRDemo:
    def __init__(self, host="localhost", port=5000):
        self.console = Console()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.running = False

        # Connect to ASR service
        try:
            self.socket.connect((host, port))
            self.console.print("[green]Connected to ASR service[/green]")
        except ConnectionRefusedError:
            self.console.print(
                "[red]Could not connect to ASR service. Make sure it's running.[/red]"
            )
            raise

    def start(self):
        """Start receiving transcriptions"""
        self.running = True

        # Send start command to ASR service
        start_command = {"type": "start"}
        self.socket.send(json.dumps(start_command).encode("utf-8"))

        self.console.print("\n[bold green]Started ASR Demo![/bold green]")
        self.console.print("Speak into your microphone. Press Ctrl+C to stop.\n")

        # Listen for transcriptions
        while self.running:
            try:
                data = self.socket.recv(1024).decode("utf-8")
                if not data:
                    break

                message = json.loads(data)
                if message["type"] == "transcription":
                    self.console.print(
                        "\n[bold cyan]Transcribed:[/bold cyan]", message["text"]
                    )
            except Exception as e:
                if self.running:
                    self.console.print(f"[red]Error: {e}[/red]")
                break

    def stop(self):
        """Stop the demo"""
        self.running = False

        # Send stop command to ASR service
        stop_command = {"type": "stop"}
        try:
            self.socket.send(json.dumps(stop_command).encode("utf-8"))
        except:
            pass

        # Close connection
        self.socket.close()
        self.console.print("\n[yellow]Demo ended.[/yellow]")


def main():
    demo = ASRDemo()
    try:
        demo.start()
    except KeyboardInterrupt:
        demo.stop()


if __name__ == "__main__":
    main()
