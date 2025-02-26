# Interview Transcript Synthesizer

## Overview
This Python script analyzes interview transcripts stored in text and Word files (.txt, .docx, .doc). It leverages the OpenAI API to extract key points, notable quotes, and emerging themes from each transcript. The script automatically handles long transcripts by splitting them into manageable chunks and then synthesizes the analyses from multiple transcripts into a comprehensive summary document. The final output can be saved in either Markdown (.md) or Word (.docx) format.

## Prerequisites
- **Python 3:** Ensure Python 3 is installed on your system.
- **Visual Studio Code (VS Code):** A beginner-friendly code editor.
- **OpenAI API Key:** Obtain an API key from [OpenAI](https://openai.com).
- **Required Python Libraries:**
  - python-docx (for reading and writing Word documents)
  - openai
  - tiktoken
  - tqdm
  - python-dotenv
  - argparse (standard library)
- **Basic Command Line Skills:** Familiarity with running scripts from the terminal.

## Installation

1. **Download or Clone the Repository:**
   - If you’re new to GitHub, click the "Code" button on the repository page and select "Download ZIP". Extract the ZIP file to a folder on your computer.

2. **Open the Project in VS Code:**
   - Launch VS Code.
   - Go to **File > Open Folder** and select the folder where you extracted or cloned the repository.

3. **Set Up a Virtual Environment (Optional but Recommended):**
   - Open the integrated terminal in VS Code (**View > Terminal**).
   - Create a virtual environment by running:
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

4. **Install Required Libraries:**
   - In the terminal, run:
     ```
     pip install -r requirements.txt
     ```
   - If a `requirements.txt` file isn’t provided, install the libraries individually:
     ```
     pip install python-docx openai tiktoken tqdm python-dotenv
     ```

## Setup

1. **Configure Environment Variables:**
   - Create a file named `.env` in the project folder.
   - Add your OpenAI API key in the following format:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - Replace `your_api_key_here` with your actual API key.

## How to Run the Script

1. **Interactive Mode:**
   - Open the integrated terminal in VS Code.
   - Run the script by typing:
     ```
     python script_name.py
     ```
     Replace `script_name.py` with the actual name of the Python file.
   - Follow the prompts to:
     - Enter the folder path containing your interview transcripts.
     - Specify an output file path (optional).
     - Choose the OpenAI model from the available options.
     - Select the desired output format (Markdown or Word).

2. **Command-Line Arguments:**
   - Alternatively, run the script with arguments:
     ```
     python script_name.py path_to_transcripts_folder --output output_file.md --model gpt-4 --format md
     ```
     Replace `script_name.py`, `path_to_transcripts_folder`, and other parameters as needed.

## Key Functionalities

- **File Reading:**
  - Reads interview transcripts from `.txt`, `.docx`, and `.doc` files.
- **Transcript Analysis:**
  - Uses the OpenAI API to analyze transcripts and extract:
    - A concise summary
    - Key points
    - Notable quotes with proper attribution
    - Emerging themes
  - Automatically splits long transcripts into chunks based on token limits.
- **Synthesis:**
  - Combines individual transcript analyses into an overall synthesis, including an executive summary, major themes, and recommendations.
- **Output Options:**
  - Saves the final synthesis document in Markdown (.md) or Word (.docx) format.
- **Progress Monitoring:**
  - Utilizes tqdm to display processing progress, making it easy to monitor batch processing.

## Troubleshooting

- **Missing API Key:**  
  Ensure your `.env` file is properly set up with a valid `OPENAI_API_KEY`.

- **Unsupported File Format:**  
  Verify that your transcript files have the correct extensions (.txt, .docx, or .doc).

- **Error Messages:**  
  Check the terminal output for error messages. Ensure file paths are correct and your internet connection is active for API calls.

## Conclusion
This beginner-friendly Interview Transcript Synthesizer automates the process of analyzing interview transcripts using advanced AI. It extracts critical insights and compiles them into a comprehensive, easy-to-read synthesis document. Customize and extend the tool as needed to suit your analysis workflow!
