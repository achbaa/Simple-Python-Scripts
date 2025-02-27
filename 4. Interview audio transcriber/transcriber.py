from openai import OpenAI
import os
import sys
import tempfile
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime
import time
import shutil
import asyncio
import aiohttp
from openai import AsyncOpenAI
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found in environment variables.")
    print("Please create a .env file with your OpenAI API key or set it as an environment variable.")
    sys.exit(1)

def get_ffmpeg_path():
    """Get the path to FFmpeg executable."""
    try:
        if os.name == 'nt':  # Windows
            ffmpeg_path = subprocess.check_output(['where', 'ffmpeg']).decode().strip()
        else:  # Linux and Mac
            ffmpeg_path = subprocess.check_output(['which', 'ffmpeg']).decode().strip()
        print(f"FFmpeg found at: {ffmpeg_path}")
        return ffmpeg_path
    except subprocess.CalledProcessError:
        print('FFmpeg not found. Please ensure it is installed and added to the PATH.')
        raise FileNotFoundError('FFmpeg not found')

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        # Check for ffmpeg
        get_ffmpeg_path()
        
        # Check for ffprobe
        if os.name == 'nt':  # Windows
            subprocess.check_output(['where', 'ffprobe'])
        else:  # Linux and Mac
            subprocess.check_output(['which', 'ffprobe'])
        
        print("All dependencies found.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Required dependencies not found. Please ensure ffmpeg and ffprobe are installed.")
        return False

def preprocess_audio(input_file, output_format="ogg", max_size_mb=24):
    """Preprocess audio file for better transcription."""
    try:
        print(f"Starting preprocessing of file: {input_file}")
        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as temp_file:
            output_file = temp_file.name
        
        # Added -y flag to force overwrite without prompting
        command = [
            "ffmpeg",
            "-y",  # Force overwrite without asking
            "-i", input_file,
            "-vn",  # No video
            "-ac", "1",  # Mono audio
            "-ar", "44100",  # 44.1kHz sample rate
            "-q:a", "2",  # Quality setting for audio
            output_file
        ]
        
        print(f"Executing FFmpeg command: {' '.join(command)}")
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )
        
        if process.returncode != 0:
            print(f"FFmpeg error: {process.stderr}")
            return None
            
        # Check if the output file is still too large
        output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        if output_size_mb > max_size_mb:
            print(f"Processed file still too large ({output_size_mb:.2f} MB). Applying additional compression.")
            
            # Create another temporary file for the more compressed version
            with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as compressed_temp:
                compressed_output = compressed_temp.name
            
            # Apply more aggressive compression
            compress_command = [
                "ffmpeg",
                "-y",
                "-i", output_file,
                "-vn",
                "-ac", "1",
                "-ar", "32000",  # Lower sample rate
                "-b:a", "64k",   # Lower bitrate
                compressed_output
            ]
            
            print(f"Executing compression command: {' '.join(compress_command)}")
            compress_process = subprocess.run(
                compress_command,
                capture_output=True,
                text=True,
                check=False
            )
            
            # Delete the first temporary file
            Path(output_file).unlink(missing_ok=True)
            
            if compress_process.returncode != 0:
                print(f"FFmpeg compression error: {compress_process.stderr}")
                return None
                
            output_file = compressed_output
            
        print(f"Preprocessing completed. Output file: {output_file}")
        return output_file
    except Exception as e:
        print(f"Unexpected error during preprocessing: {str(e)}")
        return None

def get_audio_duration(file_path):
    """Get duration of audio file in minutes."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        output = subprocess.check_output(cmd).decode().strip()
        duration = float(output) / 60  # Convert seconds to minutes
        print(f"Audio duration: {duration:.2f} minutes")
        return duration
    except Exception as e:
        print(f"Error getting audio duration: {str(e)}")
        return 0

def get_audio_file_size(file_path):
    """Get file size in MB."""
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        print(f"Audio file size: {size_mb:.2f} MB")
        return size_mb
    except Exception as e:
        print(f"Error getting file size: {str(e)}")
        return 0

def split_long_audio(file_path, max_duration_minutes=25):
    """Split long audio files into chunks of specified maximum duration."""
    try:
        duration = get_audio_duration(file_path)
        if duration <= max_duration_minutes:
            return [file_path]  # No need to split
            
        print(f"Splitting audio file of {duration:.2f} minutes into {max_duration_minutes}-minute chunks")
        
        # Create a temporary directory for the chunks
        temp_dir = tempfile.mkdtemp()
        file_extension = os.path.splitext(file_path)[1]
        base_name = os.path.basename(file_path).replace(file_extension, '')
        
        # Calculate number of chunks needed
        num_chunks = int(duration / max_duration_minutes) + 1
        
        # Split the audio file
        chunk_files = []
        for i in range(num_chunks):
            start_time = i * max_duration_minutes * 60  # Convert to seconds
            output_file = os.path.join(temp_dir, f"{base_name}_part{i+1}{file_extension}")
            
            command = [
                "ffmpeg",
                "-y",
                "-i", file_path,
                "-ss", str(start_time),
                "-t", str(max_duration_minutes * 60),  # Duration in seconds
                "-c", "copy",  # Copy without re-encoding
                output_file
            ]
            
            print(f"Executing split command for chunk {i+1}/{num_chunks}")
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False
            )
            
            if process.returncode != 0:
                print(f"Error splitting chunk {i+1}: {process.stderr}")
                continue
                
            chunk_files.append(output_file)
            
        print(f"Split audio into {len(chunk_files)} chunks")
        return chunk_files
    except Exception as e:
        print(f"Error splitting audio file: {str(e)}")
        return [file_path]  # Return original file on error

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def transcribe_with_whisper(file_path, model="whisper-1", max_retries=3):
    """Transcribe an audio file using OpenAI's Whisper model."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        print(f"Starting transcription of file: {file_path}")
        
        # Check file size and duration
        file_size = get_audio_file_size(file_path)
        duration = get_audio_duration(file_path)
        
        # Process file if needed (OpenAI has a 25MB limit)
        processed_file = file_path
        temp_files_to_delete = []
        
        # If file is too long, split it
        if duration > 25:
            print(f"Audio duration ({duration:.2f} min) exceeds limit. Splitting into chunks.")
            chunk_files = split_long_audio(file_path)
            
            if len(chunk_files) > 1:
                # Transcribe each chunk and combine
                all_transcriptions = []
                
                # Process chunks concurrently with semaphore for rate limiting
                semaphore = asyncio.Semaphore(4)  # Limit concurrent API calls
                
                async def process_chunk(chunk_file, chunk_index):
                    async with semaphore:
                        print(f"Transcribing chunk {chunk_index+1}/{len(chunk_files)}")
                        
                        # Check if chunk needs preprocessing
                        chunk_size = get_audio_file_size(chunk_file)
                        if chunk_size > 24:
                            processed_chunk = preprocess_audio(chunk_file)
                            if not processed_chunk:
                                return f"[Failed to preprocess chunk {chunk_index+1}]"
                            temp_files_to_delete.append(processed_chunk)
                        else:
                            processed_chunk = chunk_file
                        
                        # Transcribe with retries
                        for attempt in range(max_retries):
                            try:
                                with open(processed_chunk, "rb") as audio_file:
                                    chunk_transcription = await client.audio.transcriptions.create(
                                        model=model,
                                        file=audio_file,
                                        response_format="text"
                                    )
                                return chunk_transcription
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    print(f"Attempt {attempt+1} failed: {str(e)}. Retrying in 5 seconds...")
                                    await asyncio.sleep(5)
                                else:
                                    return f"[Transcription failed for part {chunk_index+1}]"
                
                # Process all chunks concurrently
                tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunk_files)]
                all_transcriptions = await asyncio.gather(*tasks)
                
                # Clean up temporary chunk files
                for chunk_file in chunk_files:
                    if chunk_file != file_path:  # Don't delete the original file
                        Path(chunk_file).unlink(missing_ok=True)
                
                # Clean up the temp directory
                if len(chunk_files) > 1 and os.path.dirname(chunk_files[0]) != os.path.dirname(file_path):
                    shutil.rmtree(os.path.dirname(chunk_files[0]), ignore_errors=True)
                
                # Combine all transcriptions
                transcription = "\n\n".join(all_transcriptions)
                
                # Clean up any other temp files
                for temp_file in temp_files_to_delete:
                    Path(temp_file).unlink(missing_ok=True)
                
                print("Combined transcription of all chunks completed successfully")
                return transcription
        
        # If file is too large but not too long, preprocess it
        if file_size > 24:
            print(f"File size: {file_size:.2f} MB exceeds limit. Preprocessing required.")
            processed_file = preprocess_audio(file_path)
            if not processed_file:
                raise Exception("Audio preprocessing failed")
            temp_files_to_delete.append(processed_file)
        
        # Transcribe with retries
        for attempt in range(max_retries):
            try:
                # Transcribe the audio
                with open(processed_file, "rb") as audio_file:
                    transcription = await client.audio.transcriptions.create(
                        model=model,
                        file=audio_file,
                        response_format="text"
                    )
                
                # Clean up temporary files
                for temp_file in temp_files_to_delete:
                    Path(temp_file).unlink(missing_ok=True)
                
                print("Transcription completed successfully")
                return transcription
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt+1} failed: {str(e)}. Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    # Clean up temporary files on final failure
                    for temp_file in temp_files_to_delete:
                        Path(temp_file).unlink(missing_ok=True)
                    print(f"All {max_retries} attempts failed: {str(e)}")
                    return f"Transcription error: {str(e)}"
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return f"Transcription error: {str(e)}"

async def summarize_transcript(transcript, model="gpt-4o-mini", max_retries=3):
    """Summarize the transcript using OpenAI's chat completions."""
    try:
        if not transcript or transcript.startswith("Transcription error"):
            return "Cannot summarize due to transcription error."
            
        print("Starting transcript summarization")
        
        system_prompt = """You are a highly skilled assistant specialized in analyzing and summarizing interview transcripts.
        Please provide a comprehensive summary that includes:
        1. Main topics and key points discussed
        2. Important insights or opinions expressed
        3. Notable quotes (with attribution if possible)
        4. Key themes that emerged in the interview
        
        Format the summary with clear sections and bullet points for readability."""
        
        # Try with retries
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Please analyze and summarize this interview transcript:\n\n{transcript}"}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                
                summary = response.choices[0].message.content
                print("Summary generated successfully")
                return summary
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt+1} failed: {str(e)}. Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    print(f"All {max_retries} attempts failed: {str(e)}")
                    return f"Summarization error: {str(e)}"
    except Exception as e:
        print(f"Error during summarization: {str(e)}")
        return f"Summarization error: {str(e)}"

async def process_audio_file(file_path, output_dir=None, transcription_model="whisper-1", summary_model="gpt-4o-mini", create_summary=True):
    """Process a single audio file: transcribe and optionally summarize."""
    try:
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        
        # Set default output directory if not specified
        if output_dir is None:
            # Use a path relative to the script location
            script_dir = Path(__file__).parent
            output_dir = script_dir / "Output files"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing file: {file_name}")
        
        # Step 1: Transcription
        transcript = await transcribe_with_whisper(file_path, transcription_model)
        if transcript.startswith("Transcription error"):
            print(f"Transcription failed for {file_name}: {transcript}")
            
            # Save error message to file
            error_path = os.path.join(output_dir, f"{base_name}_error.txt")
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"Error processing {file_name}:\n{transcript}")
            
            return False
        
        # Save transcript
        transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        
        print(f"Transcript saved to {transcript_path}")
        
        # Step 2: Summarization (if requested)
        if create_summary:
            summary = await summarize_transcript(transcript, summary_model)
            if summary.startswith("Summarization error"):
                print(f"Summarization failed for {file_name}: {summary}")
                
                # Still save the transcript without summary
                combined_path = os.path.join(output_dir, f"{base_name}_full.txt")
                with open(combined_path, "w", encoding="utf-8") as f:
                    f.write("TRANSCRIPT\n")
                    f.write("="*50 + "\n")
                    f.write(transcript)
                    f.write("\n\nSUMMARY\n")
                    f.write("="*50 + "\n")
                    f.write(f"Summary generation failed: {summary}")
            else:
                # Save summary
                summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                print(f"Summary saved to {summary_path}")
                
                # Save combined file
                combined_path = os.path.join(output_dir, f"{base_name}_full.txt")
                with open(combined_path, "w", encoding="utf-8") as f:
                    f.write("TRANSCRIPT\n")
                    f.write("="*50 + "\n")
                    f.write(transcript)
                    f.write("\n\nSUMMARY\n")
                    f.write("="*50 + "\n")
                    f.write(summary)
                print(f"Combined transcript and summary saved to {combined_path}")
        
        return True
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        
        # Try to save error message
        try:
            os.makedirs(output_dir, exist_ok=True)
            error_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_error.txt")
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"Error processing {os.path.basename(file_path)}:\n{str(e)}")
        except:
            pass
            
        return False

async def process_audio_folder(input_dir, output_dir=None, transcription_model="whisper-1", summary_model="gpt-4o-mini", create_summary=True):
    """Process all audio files in a folder."""
    try:
        # Validate input directory
        if not os.path.isdir(input_dir):
            print(f"Input directory does not exist: {input_dir}")
            return False
        
        # Set default output directory if not specified
        if output_dir is None:
            output_dir = os.path.join(input_dir, "transcripts")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all audio files
        audio_extensions = ('.mp3', '.mp4', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma')
        audio_files = []
        
        for file in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file)
            if os.path.isfile(file_path) and file.lower().endswith(audio_extensions):
                audio_files.append(file_path)
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return False
        
        # Sort files to ensure consistent processing order
        audio_files.sort()
        
        print(f"Found {len(audio_files)} audio files to process")
        print(f"Found {len(audio_files)} audio files to process:")
        for i, file_path in enumerate(audio_files, 1):
            print(f"  {i}. {os.path.basename(file_path)}")
        
        # Process files concurrently with rate limiting
        semaphore = asyncio.Semaphore(4)  # Limit concurrent file processing
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                return await process_audio_file(
                    file_path, 
                    output_dir, 
                    transcription_model, 
                    summary_model, 
                    create_summary
                )
        
        # Process all files concurrently
        tasks = [process_with_semaphore(file_path) for file_path in audio_files]
        results = await asyncio.gather(*tasks)
        
        successful = sum(1 for result in results if result)
        failed = len(results) - successful
        
        # Create a report
        report_path = os.path.join(output_dir, "transcription_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Transcription Report\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total files processed: {len(audio_files)}\n")
            f.write(f"Successfully processed: {successful}\n")
            f.write(f"Failed: {failed}\n\n")
            f.write(f"Files processed:\n")
            for file_path in audio_files:
                f.write(f"- {os.path.basename(file_path)}\n")
        
        print(f"Processing complete. Successful: {successful}, Failed: {failed}")
        print(f"Report saved to {report_path}")
        
        return True
    except Exception as e:
        print(f"Error processing folder {input_dir}: {str(e)}")
        return False

def get_user_input():
    """Get user input for processing parameters."""
    print("\nAudio Transcription and Summarization Tool")
    print("=========================================")
    
    # Get input directory
    input_dir = input("\nEnter the path to the folder containing audio files: ").strip()
    input_dir = input_dir.replace('"', '').replace("'", "")  # Remove quotes if present
    
    # Set default output directory relative to the script location
    script_dir = Path(__file__).parent
    default_output_dir = script_dir / "Output files"
    
    # Get output directory
    output_prompt = f"\nEnter the path to save output files (default: {default_output_dir}): "
    output_dir = input(output_prompt).strip()
    if not output_dir:
        output_dir = default_output_dir
    else:
        # Convert to string before using string methods
        output_dir = str(output_dir).replace('"', '').replace("'", "")  # Remove quotes if present
    
    # Get transcription model
    transcription_models = ["whisper-1"]
    print("\nAvailable transcription models:")
    for i, model in enumerate(transcription_models, 1):
        print(f"{i}. {model}")
    
    transcription_model = transcription_models[0]  # Default to first model
    
    # Get summary model
    summary_models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
    print("\nAvailable summary models:")
    for i, model in enumerate(summary_models, 1):
        print(f"{i}. {model}")
    
    while True:
        model_choice = input(f"Select a summary model (1-{len(summary_models)}, default is 1): ").strip()
        if not model_choice:
            summary_model = summary_models[0]
            break
        try:
            model_index = int(model_choice) - 1
            if 0 <= model_index < len(summary_models):
                summary_model = summary_models[model_index]
                break
            else:
                print(f"Please enter a number between 1 and {len(summary_models)}.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Ask if user wants summaries
    while True:
        create_summary = input("Generate summaries for transcripts? (y/n, default is y): ").strip().lower()
        if not create_summary or create_summary == 'y':
            create_summary = True
            break
        elif create_summary == 'n':
            create_summary = False
            break
        else:
            print("Please enter 'y' or 'n'.")
    
    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "transcription_model": transcription_model,
        "summary_model": summary_model,
        "create_summary": create_summary
    }

async def async_main():
    """Async main function to run the script."""
    # Check for dependencies
    if not check_dependencies():
        print("Required dependencies not found. Please install ffmpeg and ffprobe.")
        sys.exit(1)
    
    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        # Use argparse for command-line arguments
        parser = argparse.ArgumentParser(description='Transcribe audio files and generate summaries.')
        parser.add_argument('input_dir', help='Path to the folder containing audio files')
        parser.add_argument('--output', help='Path to save the output files', default=None)
        parser.add_argument('--transcription-model', help='OpenAI model to use for transcription', default="whisper-1")
        parser.add_argument('--summary-model', help='OpenAI model to use for summarization', default="gpt-4o-mini")
        parser.add_argument('--no-summary', help='Skip generating summaries', action='store_true')
        parser.add_argument('--single-file', help='Process a single file instead of a directory', action='store_true')
        
        args = parser.parse_args()
        input_dir = args.input_dir
        output_dir = args.output
        transcription_model = args.transcription_model
        summary_model = args.summary_model
        create_summary = not args.no_summary
        
        # Handle single file mode
        if args.single_file:
            if not os.path.isfile(input_dir):
                print(f"Error: '{input_dir}' is not a valid file.")
                sys.exit(1)
                
            if output_dir is None:
                # Use default output directory
                script_dir = Path(__file__).parent
                output_dir = script_dir / "Output files"
                
            success = await process_audio_file(
                input_dir,
                output_dir,
                transcription_model,
                summary_model,
                create_summary
            )
            
            if success:
                print(f"\nProcessing complete! Results saved to {output_dir}")
            else:
                print("\nProcessing failed. Check the log file for details.")
                
            sys.exit(0)
    else:
        # Get input interactively
        user_input = get_user_input()
        input_dir = user_input["input_dir"]
        output_dir = user_input["output_dir"]
        transcription_model = user_input["transcription_model"]
        summary_model = user_input["summary_model"]
        create_summary = user_input["create_summary"]
    
    # Process the folder
    success = await process_audio_folder(
        input_dir,
        output_dir,
        transcription_model,
        summary_model,
        create_summary
    )
    
    if success:
        print(f"\nProcessing complete! Results saved to {output_dir}")
    else:
        print("\nProcessing failed. Check the log file for details.")

if __name__ == "__main__":
    asyncio.run(async_main())
