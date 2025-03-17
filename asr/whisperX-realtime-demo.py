import whisperx
import pyaudio
import wave
import numpy as np
import threading
import queue
import time
import gc

class RealTimeTranscriber:
    def __init__(self, device="cpu", model_name="tiny.en", compute_type="int8"):
        self.device = device
        self.audio_queue = queue.Queue()
        self.running = False
        
        # Initialize whisperX model
        self.model = whisperx.load_model(model_name, device, compute_type=compute_type)
        
        # Audio recording parameters
        self.CHUNK = 1024 * 3
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
    def start_recording(self):
        self.running = True
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record_audio)
        self.record_thread.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_audio)
        self.process_thread.start()
        
    def _record_audio(self):
        while self.running:
            try:
                data = self.stream.read(self.CHUNK)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                self.audio_queue.put(audio_chunk)
            except Exception as e:
                print(f"Error recording audio: {e}")
                break
                
    def _process_audio(self):
        audio_buffer = []
        while self.running:
            # Process audio in 3-second chunks
            if len(audio_buffer) * self.CHUNK / self.RATE >= 3.0:
                try:
                    # Convert buffer to numpy array
                    audio_data = np.concatenate(audio_buffer)
                    
                    # Transcribe with whisperX
                    result = self.model.transcribe(audio_data, batch_size=1)
                    
                    # Align whisper output
                    model_a, metadata = whisperx.load_align_model(
                        language_code=result["language"], 
                        device=self.device
                    )
                    result = whisperx.align(
                        result["segments"], 
                        model_a, 
                        metadata, 
                        audio_data, 
                        self.device, 
                        return_char_alignments=False
                    )
                    
                    # Print results
                    if result["segments"]:
                        print("\nTranscription:", result["segments"][0]["text"])
                    
                    # Clear buffer
                    audio_buffer = []
                    
                    # Clean up alignment model
                    del model_a
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing audio: {e}")
                    audio_buffer = []
            
            # Add new audio to buffer
            if not self.audio_queue.empty():
                audio_buffer.append(self.audio_queue.get())
                
    def stop(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.record_thread.join()
        self.process_thread.join()

# Usage example
if __name__ == "__main__":
    transcriber = RealTimeTranscriber(device="cpu",model_name="tiny.en")
    print("Starting real-time transcription... Press Ctrl+C to stop")
    try:
        transcriber.start_recording()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping transcription...")
        transcriber.stop()