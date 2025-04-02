import signal
import sys
import argparse
from client import Client
from audio_streamer import AudioStreamer
import logging
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def receive_transcripts(streamer):
    """Receive and display transcripts from the server."""
    while True:
        try:
            data = streamer.socket.recv(4096)
            if not data:
                break
            transcript = data.decode()
            if transcript:
                print(f"\nTranscription: {transcript}\n")
        except Exception as e:
            logger.error(f"Error receiving transcript: {e}")
            break


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    logger.info("Stopping application...")
    if client:
        client.stop()
    if streamer:
        streamer.stop()
    sys.exit(0)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Speech-to-Text Client")
    parser.add_argument("--host", default="localhost", help="Server host address")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    args = parser.parse_args()

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize components
    client = Client()
    streamer = AudioStreamer(host=args.host, port=args.port)

    try:
        # Start recording
        logger.info("Starting audio recording...")
        client.start_recording()

        # Start streaming
        logger.info(f"Starting audio streaming to {args.host}:{args.port}...")
        streamer.start_streaming()

        # Start transcript receiver thread
        receiver_thread = threading.Thread(target=receive_transcripts, args=(streamer,))
        receiver_thread.daemon = True
        receiver_thread.start()

        # Main loop
        while True:
            # Get audio chunk from client
            audio_chunk = client.audio_queue.get()

            # Send chunk to server
            streamer.send_chunk(audio_chunk)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        # Clean up
        client.stop()
        streamer.stop()
