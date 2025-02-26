# Audio Transcription and Summarization Tool

## Overview
This script transcribes audio files and, optionally, generates summaries using OpenAI's models. It:
- Preprocesses audio files (compression, splitting) using FFmpeg.
- Transcribes audio with OpenAI's Whisper model.
- Optionally summarizes transcripts using an OpenAI summarization model.
- Supports processing a single file or an entire folder.
- Generates output files including transcripts, summaries, and a combined report.

## Prerequisites
- **Python 3:** Ensure Python 3 is installed on your system.
- **FFmpeg & FFprobe:** Install both and add them to your PATH.
- **OpenAI API Key:** Obtain your API key from [OpenAI](https://openai.com) and add it to a `.env` file.
- **Visual Studio Code (VS Code):** Recommended for beginners.
- **Required Python Libraries:**
  - openai
  - python-dotenv
  - tqdm
  - argparse (standard library)
  - shutil, logging, subprocess, tempfile, sys, datetime (standard libraries)

## Installation

1. **Download or Clone the Repository:**
   - If you're new to GitHub, click the "Code" button on the repository page and select "Download ZIP". Extract the ZIP file.

2. **Open the Project in VS Code:**
   - Launch VS Code.
   - Go to **File > Open Folder** and select the project folder.

3. **Set Up a Virtual Environment (Optional but Recommended):**
   - Open the integrated terminal in VS Code.
   - Run:
     ```
     python -m venv env
     ```
   - Activate the virtual environment:
     - **Windows:**
       ```
       .\env\Scripts\activate
       ```
     - **macOS/Linux:**
       ```
       source env/bin/activate
       ```

4. **Install Dependencies:**
   - Run:
     ```
     pip install -r requirements.txt
     ```
   - If no `requirements.txt` is provided, install libraries individually:
     ```
     pip install openai python-dotenv tqdm
     ```

## Setup

1. **Configure Environment Variables:**
   - Create a file named `.env` in the project folder.
   - Add your OpenAI API key in the following format:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - Replace `your_api_key_here` with your actual API key.

2. **Ensure FFmpeg and FFprobe are Installed:**
   - Verify that FFmpeg and FFprobe are installed and accessible from the command line.

## FFmpeg Installation Guide

This tool requires FFmpeg and FFprobe to process audio files. Here's how to install them:

### Windows
1. **Download FFmpeg**:
   - Go to [FFmpeg.org](https://ffmpeg.org/download.html) or [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) for Windows builds
   - Download the "essentials" or "full" build (7z or zip archive)

2. **Extract the Archive**:
   - Extract the downloaded archive to a location like `C:\ffmpeg`

3. **Add to PATH**:
   - Right-click on "This PC" or "My Computer" and select "Properties"
   - Click on "Advanced system settings"
   - Click the "Environment Variables" button
   - Under "System variables", find and select "Path", then click "Edit"
   - Click "New" and add the path to the FFmpeg bin folder (e.g., `C:\ffmpeg\bin`)
   - Click "OK" on all dialogs to save changes

4. **Verify Installation**:
   - Open a new Command Prompt window
   - Type `ffmpeg -version` and press Enter
   - You should see version information if FFmpeg is installed correctly

### macOS
1. **Using Homebrew** (recommended):
   - Install Homebrew if you don't have it: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
   - Install FFmpeg: `brew install ffmpeg`

2. **Verify Installation**:
   - Open Terminal
   - Type `ffmpeg -version` and press Enter
   - You should see version information if FFmpeg is installed correctly

### Linux
1. **Ubuntu/Debian**:
   ```
   sudo apt update
   sudo apt install ffmpeg
   ```

2. **Fedora**:
   ```
   sudo dnf install ffmpeg
   ```

3. **Verify Installation**:
   - Open Terminal
   - Type `ffmpeg -version` and press Enter
   - You should see version information if FFmpeg is installed correctly

## How to Run the Script

1. **Interactive Mode:**
   - Open the integrated terminal in VS Code.
   - Run the script:
     ```
     python script_name.py
     ```
     Replace `script_name.py` with the actual name of the Python file.
   - Follow the prompts to:
     - Enter the folder containing your audio files.
     - Specify an output folder (or use the default).
     - Choose the transcription and summarization models.
     - Decide whether to generate summaries.

2. **Command-Line Mode:**
   - Run the script with arguments:
     ```
     python script_name.py input_folder_path --output output_folder_path --transcription-model whisper-1 --summary-model gpt-4o-mini
     ```
   - Additional flags:
     - `--no-summary` to skip generating summaries.
     - `--single-file` to process a single audio file instead of a folder.

## Key Functionalities

- **Dependency Check:**
  - Verifies FFmpeg and FFprobe are installed.
- **Audio Preprocessing:**
  - Converts audio to a supported format.
  - Compresses and splits long audio files into smaller chunks if necessary.
- **Transcription:**
  - Uses OpenAI's Whisper model with retry logic for robust transcription.
- **Summarization:**
  - Optionally generates summaries using an OpenAI model.
- **Output Generation:**
  - Saves transcripts, summaries, and a combined report in the specified output folder.
  - Generates a transcription report listing processed files and statuses.
- **Logging & Progress Tracking:**
  - Detailed logging of steps and errors (saved in `transcription.log`).
  - Progress is displayed via tqdm.

## Troubleshooting

- **Missing API Key:**  
  Ensure your `.env` file contains a valid `OPENAI_API_KEY`.

- **Dependency Issues:**  
  Verify that FFmpeg and FFprobe are installed and properly added to your PATH.

- **File Errors:**  
  Make sure the input directory contains supported audio file formats (e.g., .mp3, .wav, .m4a, etc.).

- **Transcription/Summarization Failures:**  
  Check the `transcription.log` file for detailed error messages and ensure your internet connection is active.

## Conclusion
This Audio Transcription and Summarization Tool is designed for beginners to easily transcribe and summarize audio files using advanced AI. With both interactive prompts and command-line options, the tool automates audio processing tasks—making it ideal for interviews, podcasts, lectures, and more. Customize and extend the script to fit your specific needs!
