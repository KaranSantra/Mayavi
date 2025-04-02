import signal
import sys
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
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize components
    client = Client()
    streamer = AudioStreamer()

    try:
        # Start recording
        logger.info("Starting audio recording...")
        client.start_recording()

        # Start streaming
        logger.info("Starting audio streaming...")
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
