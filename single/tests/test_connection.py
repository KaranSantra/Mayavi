import socket
import sys
import time


def test_port(host, port):
    print(f"\nTesting port {port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)  # 5 second timeout

    try:
        print(f"Attempting to connect to {host}:{port}")
        start_time = time.time()
        sock.connect((host, port))
        end_time = time.time()
        print(f"Successfully connected to server on port {port}!")
        print(f"Connection time: {end_time - start_time:.2f} seconds")
        return True
    except socket.timeout:
        print(f"Connection timed out on port {port}")
        return False
    except ConnectionRefusedError:
        print(f"Connection refused on port {port} - server might not be running")
        return False
    except Exception as e:
        print(f"Error on port {port}: {str(e)}")
        return False
    finally:
        sock.close()


def test_connection():
    host = "35.192.131.234"
    ports_to_try = [5000, 5001, 8000, 8080]  # Common ports to try

    print(f"Starting connection tests to {host}")

    for port in ports_to_try:
        if test_port(host, port):
            return True

    print("\nAll connection attempts failed")
    return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
