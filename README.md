# Install required packages

pip install -r requirements.txt

# Start the server (it will listen on all interfaces)

python server.py

# Run the client, replacing VM_IP with your Cloud VM's IP address

python main.py --host VM_IP --port 5000
