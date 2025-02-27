# Flexible Data Manipulator Tool

## Overview
This Python script uses OpenAI's API to transform data in an Excel or CSV file. It reads specified input columns from your file, applies custom transformation instructions (provided as a prompt), and writes the transformed results to new output columns. The tool processes data in batches using parallel processing for efficiency and logs all its activities for troubleshooting.

## Prerequisites
- **Python 3.7+:** The script uses asyncio features that require Python 3.7 or newer.
- **Visual Studio Code (VS Code):** A user-friendly code editor recommended for beginners.
- **OpenAI API Key:** Sign up at [OpenAI](https://openai.com) to obtain an API key.
- **Excel/CSV File:** An input file containing your data.
- **Required Python Libraries:**
  - pandas
  - numpy
  - openai
  - python-dotenv
  - tqdm
  - aiohttp
  - argparse
  - (Other standard libraries: os, sys, json, time, logging, datetime, asyncio)

## Installation

1. **Download or Clone the Repository:**
   - If you're new to GitHub, click the "Code" button on the repository page and select "Download ZIP". Extract the ZIP file to a folder on your computer.

2. **Open the Project in VS Code:**
   - Launch VS Code.
   - Go to **File > Open Folder** and select the folder where you extracted or cloned the repository.

3. **Set Up a Virtual Environment (Optional but Recommended):**
   - Open the integrated terminal in VS Code.
   - Create a virtual environment:
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
   - If a `requirements.txt` is not provided, install the required libraries individually:
     ```
     pip install pandas numpy openai python-dotenv tqdm aiohttp
     ```

## Setup

1. **Configure Environment Variables:**
   - Create a file named `.env` in the project folder.
   - Add your OpenAI API key in the following format:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - Replace `your_api_key_here` with your actual API key.

2. **Outputs Directory:**
   - The script automatically creates an `outputs` folder (in the same directory as the script) where the transformed Excel file and log files will be saved.

## How to Run the Script

### Interactive Mode
1. Open the integrated terminal in VS Code.
2. Run the script:

python script_name.py

Replace `script_name.py` with the actual filename.
3. Follow the prompts:
- **Input File:** Provide the path to your Excel or CSV file.
- **Select Input Columns:** Choose which columns to use as input (by entering the corresponding numbers).
- **Output Columns:** Either select existing columns or create new ones for the transformed data.
- **Transformation Instructions:** Enter your custom instructions (e.g., "Translate the text from English to Spanish" or "Summarize the customer feedback"). Type `END` on a new line when finished.
- **Output File Path:** Specify where to save the new Excel file or use the default.
- **Model, Temperature, and Batch Size:** Choose the OpenAI model, set a temperature (randomness), and specify the batch size for processing.

### Command-Line Arguments
Alternatively, you can run the script with arguments:

python script_name.py --input path/to/input.xlsx --input-columns "Column1,Column2" --output-columns "NewCol1,NewCol2" --prompt "Your transformation instructions here" --output path/to/output.xlsx --model gpt-3.5-turbo --temperature 0.3 --batch-size 20

- **--input:** Path to the input Excel or CSV file.
- **--input-columns:** Comma-separated list of input column names.
- **--output-columns:** Comma-separated list of output column names.
- **--prompt:** Transformation instructions.
- **--output:** (Optional) Output file path.
- **--model, --temperature, --batch-size:** (Optional) Customize the API model settings.

## Key Functionalities

- **Data Reading & Validation:**  
  Reads your input file (Excel or CSV) and validates that the selected columns exist.

- **Custom Data Transformation:**  
  Uses your transformation instructions to process each row. It creates a system prompt and a batch prompt that includes sample examples for consistency.

- **Asynchronous Processing:**  
  Uses Python's asyncio for concurrent processing of data batches, significantly improving performance with large datasets. The script maintains rate limiting to avoid overwhelming the API.

- **Batch Processing:**  
  Processes data in configurable batch sizes with concurrent API calls for optimal performance.

- **Logging:**  
  Records all steps, warnings, and errors in a log file saved in the outputs directory for easy troubleshooting.

- **Output Generation:**  
  Saves the transformed data as an Excel file. A copy of the log file is also saved alongside the output.

## Performance Considerations

- **Batch Size:**  
  The default batch size is 20 rows per API call. You can adjust this based on your data characteristics and API rate limits.

- **Concurrent Processing:**  
  The script processes multiple batches concurrently while respecting API rate limits. The default maximum concurrent API calls is 4, but you can adjust this based on your API tier and requirements.

- **Memory Usage:**  
  For very large files, the script loads the entire dataset into memory. If you're processing extremely large files, consider implementing chunked reading.

## Troubleshooting

- **Missing API Key:**  
  Ensure your `.env` file contains a valid `OPENAI_API_KEY`.

- **File Issues:**  
  Confirm that the input file path is correct and that the file is in Excel or CSV format.

- **Column Validation:**  
  Make sure the columns you select exist in the file. The script will alert you if any columns are missing.

- **Processing Errors:**  
  Check the log file in the outputs directory for detailed error messages if the data transformation fails.

## Conclusion
The Flexible Data Manipulator Tool is a beginner-friendly script that leverages OpenAI's API to transform your data based on custom instructions. Whether you want to categorize text, translate content, summarize feedback, or perform any other data transformation, this tool provides a simple and efficient way to automate the process. Customize the prompts and settings to fit your needs and enjoy the power of AI-driven data manipulation!
