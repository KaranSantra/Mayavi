# llm/llm_module.py
import os
import sys
import time
import socket
import json
import threading
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMModule:
    def __init__(
        self,
        host="127.0.0.1",
        port=5001,
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        device="mps",
    ):
        # Setup socket
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False

        # Setup device
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"LLM Module: Loading Llama model on {self.device}...")

        # Load tokenizer with proper configuration
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)

        print("LLM Module: Model loaded successfully!")

        # Chat history for context
        self.history = []

    def start_server(self):
        """Start a socket server to communicate with the main program"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print(f"LLM Module: Socket server started at {self.host}:{self.port}")
            self.client_socket, addr = self.server_socket.accept()
            print(f"LLM Module: Connected to client at {addr}")
            return True
        except Exception as e:
            print(f"LLM Module: Socket error - {e}")
            return False

    def send_response(self, response):
        """Send LLM response through the socket"""
        if self.client_socket:
            try:
                message = json.dumps({"type": "llm_response", "text": response})
                self.client_socket.sendall(f"{message}\n".encode("utf-8"))
                print(f"LLM Module: Sent response")
            except Exception as e:
                print(f"LLM Module: Error sending response - {e}")

    def format_prompt(self, user_input):
        """Format the prompt with chat history and system message"""
        # Start with system message that encourages concise responses
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Be concise and conversational in your responses. Keep answers brief but informative.",
            }
        ]

        for entry in self.history[-4:]:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        # Format into Llama chat format
        formatted_prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                formatted_prompt += f"<|system|>\n{msg['content']}\n"
            elif msg["role"] == "user":
                formatted_prompt += f"<|user|>\n{msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted_prompt += f"<|assistant|>\n{msg['content']}\n"

        formatted_prompt += "<|assistant|>\n"
        return formatted_prompt

    def generate_response(self, user_input):
        """Generate a response using the model"""
        prompt = self.format_prompt(user_input)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, return_attention_mask=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,  # Reduced from 512
                temperature=0.8,  # Slightly increased for more natural responses
                top_p=0.7,  # Reduced from 0.9 for more focused responses
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.3,  # Increased slightly to reduce repetition
                do_sample=True,  # Enable sampling for more natural responses
                no_repeat_ngram_size=2,  # Prevent repeating 2-grams
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        # Add to history
        self.history.append({"user": user_input, "assistant": response})

        return response

    def process_messages(self):
        """Process incoming messages from the main program"""
        while self.running:
            try:
                # Receive data from client
                data = self.client_socket.recv(4096)
                if not data:
                    break

                # Process received data
                decoded_data = data.decode("utf-8").strip()
                messages = [json.loads(msg) for msg in decoded_data.split("\n") if msg]

                for message in messages:
                    if message.get("type") == "transcription":
                        user_input = message.get("text", "").strip()
                        if user_input:
                            print(f"LLM Module: Received transcription: {user_input}")
                            response = self.generate_response(user_input)
                            self.send_response(response)
            except json.JSONDecodeError as e:
                print(f"LLM Module: JSON decode error - {e}")
            except Exception as e:
                print(f"LLM Module: Error processing message - {e}")
                if not self.running:
                    break

    def start(self):
        """Start the LLM module"""
        if not self.start_server():
            return False

        self.running = True
        self.process_thread = threading.Thread(target=self.process_messages)
        self.process_thread.start()
        return True

    def stop(self):
        """Stop the LLM module"""
        self.running = False

        if self.client_socket:
            self.client_socket.close()

        if self.server_socket:
            self.server_socket.close()

        if hasattr(self, "process_thread") and self.process_thread:
            self.process_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Module")
    parser.add_argument("--port", type=int, default=5001, help="Port to listen on")
    args = parser.parse_args()

    llm_module = LLMModule(port=args.port)
    print(f"Starting LLM Module on port {args.port}... Press Ctrl+C to stop")
    try:
        if llm_module.start():
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping LLM Module...")
        llm_module.stop()
