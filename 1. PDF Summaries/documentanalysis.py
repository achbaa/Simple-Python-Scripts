import os
from pathlib import Path
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI
import logging
from datetime import datetime
import time
import backoff

# Set up logging to console only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    def __init__(self, custom_prompt=None):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.processed_count = 0
        self.failed_files = []
        
        # Set default prompt or use custom prompt
        self.system_prompt = "You are a helpful assistant that summarizes documents. Focus on key insights and main points."
        self.user_prompt_template = "Please summarize the following text, focusing on the main insights and key points:\n\n{text}"
        
        if custom_prompt:
            self.user_prompt_template = custom_prompt + ":\n\n{text}"
    
    def read_pdf(self, file_path):
        """Read a PDF file and extract its text content."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {str(e)}")
            return None

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_tries=5,
        max_time=300
    )
    def get_summary(self, text):
        """Get summary from OpenAI API with exponential backoff retry."""
        try:
            # Add a small delay between API calls to prevent rate limiting
            time.sleep(1)
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt_template.format(text=text)}
                ],
                max_tokens=500,
                timeout=60  # Add timeout parameter
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting summary from OpenAI: {str(e)}")
            raise  # Let backoff handle the retry

    def process_directory(self, directory_path, batch_size=50):
        """Process all PDF files in the given directory with batching."""
        directory = Path(directory_path)
        
        # Check if directory exists
        if not directory.exists():
            logger.error(f"Directory {directory_path} does not exist!")
            return None
        
        # Check if there are any PDF files
        pdf_files = list(directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return None
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Create fixed output directory for summaries
        # Get the script directory
        script_dir = Path(__file__).parent
        # Create path to the Summaries folder
        output_dir = script_dir / "Summaries"
        output_dir.mkdir(exist_ok=True)
        
        # Create main summary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_output_file = output_dir / f"summaries_{timestamp}.txt"
        failed_files_log = output_dir / f"failed_files_{timestamp}.txt"
        
        # Process files in batches
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i + batch_size]
            self._process_batch(batch, main_output_file)
            
            # Save progress after each batch
            self._save_failed_files(failed_files_log)
            
            logger.info(f"Completed batch {i//batch_size + 1}/{(len(pdf_files)-1)//batch_size + 1}")
            logger.info(f"Processed {self.processed_count}/{len(pdf_files)} files")
            
        # Final status report
        logger.info(f"Processing complete. Total files processed: {self.processed_count}")
        logger.info(f"Failed files: {len(self.failed_files)}")
        logger.info(f"Results saved to: {main_output_file}")
        
        return main_output_file

    def _process_batch(self, batch, output_file):
        """Process a batch of PDF files."""
        summaries = []
        
        for file_path in batch:
            try:
                logger.info(f"Processing {file_path.name}")
                
                # Read PDF with timeout protection
                text = self.read_pdf(file_path)
                if text is None:
                    self.failed_files.append((file_path.name, "Failed to read PDF"))
                    continue
                
                # Get summary with retry logic (handled by backoff decorator)
                summary = self.get_summary(text)
                if summary is None:
                    self.failed_files.append((file_path.name, "Failed to get summary"))
                    continue
                
                summaries.append({
                    "title": file_path.name,
                    "summary": summary
                })
                
                self.processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")
                self.failed_files.append((file_path.name, str(e)))
                continue
        
        # Write batch results to file
        self._write_summaries(summaries, output_file)

    def _write_summaries(self, summaries, output_file):
        """Write summaries to file with error handling."""
        try:
            with open(output_file, "a", encoding="utf-8") as f:
                for item in summaries:
                    f.write(f"\n{'='*50}\n")
                    f.write(f"Document: {item['title']}\n")
                    f.write(f"{'='*50}\n\n")
                    f.write(item['summary'])
                    f.write("\n\n")
        except Exception as e:
            logger.error(f"Error writing summaries to file: {str(e)}")

    def _save_failed_files(self, failed_files_log):
        """Save list of failed files to log."""
        try:
            with open(failed_files_log, "w", encoding="utf-8") as f:
                for file_name, error in self.failed_files:
                    f.write(f"{file_name}: {error}\n")
        except Exception as e:
            logger.error(f"Error writing failed files log: {str(e)}")

def main():
    # Get custom prompt from user
    print("\n=== PDF Document Analyzer ===\n")
    print("You can customize how the AI summarizes your documents.")
    custom_prompt = input("Enter your custom instructions for summarizing (press Enter for default): ").strip()
    
    # Initialize analyzer with custom prompt if provided
    analyzer = DocumentAnalyzer(custom_prompt if custom_prompt else None)
    
    # Get directory path from user
    directory = input("\nEnter the directory path containing PDF files: ").strip()
    
    # Handle both regular and raw string paths
    directory = directory.replace('"', '').replace("'", "")
    
    batch_size = input("Enter batch size (default 50): ").strip()
    batch_size = int(batch_size) if batch_size.isdigit() else 50
    
    # Process the directory
    output_file = analyzer.process_directory(directory, batch_size)
    if output_file:
        print(f"\nProcess completed. Summaries saved to: {output_file}")
    else:
        print("\nProcess failed. Please check the logs for details.")

if __name__ == "__main__":
    main()
