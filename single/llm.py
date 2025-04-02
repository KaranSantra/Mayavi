import os
import sys
import time
import socket
import json
import threading
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMModule:
    def __init__(
        self, host="127.0.0.1", port=5001, model_name="meta-llama/Llama-3.2-1B-Instruct"
    ):
        """Initialize the LLM module."""
        self.host = host
        self.port = port
        self.model_name = model_name
        self.history = []
        self.max_history = 4

        # Auto-detect device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(f"Loading model and tokenizer from {model_name} on {self.device}")

        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model with appropriate dtype based on device
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        logger.info("Model and tokenizer loaded successfully")

    def format_prompt(self, user_input):
        """Format the prompt with chat history."""
        # Start with system message that encourages concise responses
        prompt = "<|system|>\nYou are a helpful AI assistant. Be concise and conversational in your responses. Keep answers brief but informative.\n\n"

        # Add chat history
        for entry in self.history[-self.max_history :]:
            prompt += (
                f"<|user|>\n{entry['user']}\n<|assistant|>\n{entry['assistant']}\n\n"
            )

        # Add current user input
        prompt += f"<|user|>\n{user_input}\n<|assistant|>\n"

        logger.info(f"Formatted prompt: {prompt}")
        return prompt

    def generate_response(self, user_input):
        """Generate response using the language model."""
        try:
            logger.info(f"Generating response for input: {user_input}")

            # Format prompt with history
            prompt = self.format_prompt(user_input)

            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, return_attention_mask=True
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Generate response
            logger.info("Generating response from model")
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,  # Reduced from 512
                    temperature=0.8,  # Slightly increased for more natural responses
                    top_p=0.7,  # Reduced from 0.9 for more focused responses
                    pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=1.3,  # Increased slightly to reduce repetition
                    do_sample=True,  # Enable sampling for more natural responses
                    no_repeat_ngram_size=2,  # Prevent repeating 2-grams
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][input_ids.shape[1] :], skip_special_tokens=True
            )
            logger.info(f"Generated response: {response}")

            # Update history
            self.history.append({"user": user_input, "assistant": response})
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history :]
            logger.info(f"Updated history length: {len(self.history)}")

            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error processing your request."

    def stop(self):
        """Stop the LLM module."""
        pass  # Nothing to stop since we're not running a server


if __name__ == "__main__":
    llm_module = LLMModule()
    print(f"Starting LLM Module on port 5001... Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping LLM Module...")
        llm_module.stop()
