import logging
from datetime import datetime
from rich.logging import RichHandler
import os

# Network Configuration
SERVER_HOST = "34.122.21.225"  # Server listens on all interfaces
SERVER_PORT = 8765  # WebSocket port
CHUNK_SIZE = 1024
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = "int16"  # 16-bit PCM

# Logging Configurationå
LOG_DIR = "logs"
LOG_FILE = os.path.join(
    LOG_DIR, f"asr_llm_csm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

os.makedirs(LOG_DIR, exist_ok=True)


# Setup loggingååå
def setup_logging(source_name):
    logger = logging.getLogger(source_name)
    logger.setLevel(logging.DEBUG)

    # File handler with source identificationå
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler with rich formatting
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
