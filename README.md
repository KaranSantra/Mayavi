# Voice Chat Demo with ASR, LLM and CSM

This demo combines Automatic Speech Recognition (ASR), Large Language Model (LLM), and Conditional Speech Model (CSM) modules to create an interactive voice chat system.

## Setup

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Demo

### 1. Start Individual Modules

You need to run each module in separate terminal windows:

1. Start the ASR (Automatic Speech Recognition) module:

   ```bash
   python run_asr_module.py
   ```

2. Start the LLM (Large Language Model) module:

   ```bash
   python run_llm_module.py
   ```

3. Start the CSM (Conditional Speech Model) module:
   ```bash
   python run_csm_module.py
   ```

### 2. Run the Main Demo

After all modules are running, start the main demo in a new terminal:

```bash
python run_asr_llm_csm_demo.py
```

## Important Notes

- **Audio Playback Compatibility**: Currently, audio playback is only supported on macOS systems.

- **Restarting the System**: If you make any changes to the code or stop the main demo script:
  1. Stop all running modules
  2. Restart each module in the order mentioned above
  3. Run the main demo script again

## Troubleshooting

If you encounter any issues:

1. Ensure all modules are running before starting the main demo
2. Check that all required dependencies are installed correctly
3. Verify that you're using a compatible macOS system for audio playback
