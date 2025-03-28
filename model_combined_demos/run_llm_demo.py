import sys
import os
import socket
import json
import time

# Add the parent directory to sys.path to import llm_module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def connect_to_llm_server(host="127.0.0.1", port=5001, max_retries=5):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    retries = 0

    while retries < max_retries:
        try:
            client_socket.connect((host, port))
            print(f"Connected to LLM server at {host}:{port}")
            return client_socket
        except ConnectionRefusedError:
            print(
                f"Connection failed. Retrying in 2 seconds... ({retries + 1}/{max_retries})"
            )
            time.sleep(2)
            retries += 1

    raise ConnectionError("Failed to connect to LLM server")


def send_message(socket_conn, message):
    message_data = {"type": "transcription", "text": message}
    try:
        socket_conn.sendall(f"{json.dumps(message_data)}\n".encode("utf-8"))
    except Exception as e:
        print(f"Error sending message: {e}")
        return False
    return True


def receive_response(socket_conn):
    try:
        data = socket_conn.recv(4096)
        if not data:
            return None

        response_data = json.loads(data.decode("utf-8"))
        if response_data.get("type") == "llm_response":
            return response_data.get("text")
    except Exception as e:
        print(f"Error receiving response: {e}")
    return None


def main():
    try:
        # Connect to the LLM server
        client_socket = connect_to_llm_server()

        print("\nLLM Chat Interface")
        print("Type 'quit' or 'exit' to end the conversation")
        print("-" * 50)

        while True:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Check for exit commands
            if user_input.lower() in ["quit", "exit"]:
                break

            if user_input:
                # Send message to LLM server
                if not send_message(client_socket, user_input):
                    print("Failed to send message to server")
                    break

                # Get and print response
                print("\nAssistant: ", end="")
                start_time = time.time()
                response = receive_response(client_socket)
                end_time = time.time()
                if response:
                    print(response)
                    print(f"\nResponse time: {end_time - start_time:.2f} seconds")
                else:
                    print("No response received from server")

    except ConnectionError as e:
        print(f"Connection error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if "client_socket" in locals():
            client_socket.close()
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
