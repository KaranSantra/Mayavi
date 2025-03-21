import os
import sys
import time
from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import readline  # For better input handling
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.panel import Panel


class ChatDemo:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", device=None):
        # Initialize Rich console for pretty output
        self.console = Console()

        # Setup device
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.console.print(f"[yellow]Loading Llama model on {self.device}...[/yellow]")

        with self.console.status("[bold green]Loading model and tokenizer..."):
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

        self.console.print("[bold green]✓[/bold green] Model loaded successfully!")

        # Chat history for context
        self.history = []

    def format_prompt(self, user_input):
        """Format the prompt with chat history and system message"""
        # Start with system message
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

        # Add chat history
        for entry in self.history[-4:]:  # Keep last 4 exchanges for context
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

    def generate_response(self, prompt):
        """Generate a response using the model with streaming output"""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, return_attention_mask=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Setup streamer for token-by-token generation
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Generation parameters
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "streamer": streamer,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": 1.2,
        }

        # Start generation in a separate thread
        thread = Thread(target=lambda: self.model.generate(**generation_kwargs))
        thread.start()

        # Collect and yield generated text
        generated_text = ""
        for text in streamer:
            generated_text += text
            yield text

        return generated_text

    def chat(self):
        """Run the interactive chat interface"""
        self.console.print("\n[bold blue]Welcome to Llama Chat![/bold blue]")
        self.console.print(
            "Type your messages and press Enter. Type 'exit' to end the chat.\n"
        )

        while True:
            # Get user input
            try:
                user_input = input("\n[You]: ").strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Chat ended by user.[/yellow]")
                break

            if user_input.lower() in ["exit", "quit", "bye"]:
                self.console.print("[yellow]Chat ended by user.[/yellow]")
                break

            if not user_input:
                continue

            # Format prompt with history
            prompt = self.format_prompt(user_input)

            # Generate and stream response
            self.console.print("\n[Assistant]: ", end="")
            full_response = ""

            # Create a Live display for the streaming response
            with Live(refresh_per_second=4) as live:
                for token in self.generate_response(prompt):
                    full_response += token
                    # Update the display with the current response
                    live.update(
                        Panel(
                            Markdown(full_response),
                            title="Assistant",
                            border_style="blue",
                        )
                    )

            # Add to history
            self.history.append({"user": user_input, "assistant": full_response})

            # Add a visual separator
            self.console.print("\n" + "─" * 80 + "\n")


def main():
    # Enable Rich traceback for better error reporting
    from rich.traceback import install

    install(show_locals=True)

    # Create and run chat demo
    chat = ChatDemo()
    chat.chat()


if __name__ == "__main__":
    main()
