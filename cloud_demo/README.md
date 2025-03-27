# Cloud ASR-LLM-CSM Demo

This is a client-server implementation of the ASR-LLM-CSM demo that allows running the heavy processing on a cloud VM while capturing audio locally.

## Prerequisites

### Server (Cloud VM)

- Python 3.8+
- Running ASR service (port 5000)
- Running LLM service (port 5001)
- Running CSM service (port 5002)

### Client (Local Machine)

- Python 3.8+
- Working microphone
- Working speakers/headphones

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### On the Cloud VM (Server)

1. Make sure ASR, LLM, and CSM services are running
2. Start the server:

```bash
python server.py
```

The server will listen on all interfaces (0.0.0.0) on port 8765.

### On Your Local Machine (Client)

1. Start the client:

```bash
python client.py
```

By default, the client will try to connect to `localhost:8765`. To connect to your cloud VM, modify the `SERVER_HOST` in `config.py` to your VM's public IP address.

## Features

- Real-time audio streaming from local machine to cloud VM
- Beautiful console UI showing transcriptions and responses
- Comprehensive logging system
- Automatic reconnection handling
- Clean shutdown on Ctrl+C

## Logging

Logs are stored in the `logs` directory with timestamps. Each log entry includes:

- Timestamp
- Source (client/server)
- Log level
- Message

## Troubleshooting

1. If you can't connect to the server:

   - Check if the server is running
   - Verify the IP address and port in config.py
   - Check firewall settings on the cloud VM

2. If audio isn't working:

   - Check microphone permissions
   - Verify audio device selection
   - Check system audio settings

3. If services aren't responding:
   - Verify all services (ASR, LLM, CSM) are running
   - Check service logs for errors
   - Verify network connectivity between services
