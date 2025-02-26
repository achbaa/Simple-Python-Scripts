# Synthetic Data Generator Tool

## Overview
This Python script generates synthetic data using OpenAI's API. It creates a data schema (column names and example rows) based on user input—either from an Excel file, manual entry, or a natural language prompt—and then generates realistic synthetic data in batches. The generated data is saved as an Excel file, and detailed logs are stored in an outputs directory.

## Prerequisites
- **Python 3:** Ensure Python 3 is installed on your system.
- **OpenAI API Key:** Obtain your API key from [OpenAI](https://openai.com) and add it to a `.env` file.
- **Excel:** Required for reading schema definitions if using an Excel file.
- **Required Python Libraries:**
  - pandas
  - numpy
  - openai
  - python-dotenv
  - tqdm
  - argparse (standard library)
  - concurrent.futures (standard library)
  - logging, datetime, json, os, sys, time (standard libraries)
- **Basic Command Line Skills:** Familiarity with running scripts from the terminal or using interactive prompts.

## Installation

1. **Download or Clone the Repository:**
   - If you're new to GitHub, click the "Code" button on the repository page and select "Download ZIP". Extract the ZIP file to a folder on your computer.

2. **Open the Project in VS Code:**
   - Launch Visual Studio Code.
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
   - If a `requirements.txt` file is not provided, install the libraries individually:
     ```
     pip install pandas numpy openai python-dotenv tqdm
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
   - The script automatically creates an `outputs` folder (in the same directory as the script) where the generated Excel file and log files will be saved.

## How to Run the Script

### Interactive Mode
1. Open the integrated terminal in VS Code.
2. Run the script: 

python script_name.py

Replace `script_name.py` with the actual name of the Python file.
3. Follow the on-screen prompts to:
- Choose the method for specifying the data structure:
  - Provide an Excel file with column names and example rows.
  - Manually enter column names.
  - Describe the desired data in natural language (prompt), which the script will use to generate a schema.
- Enter the number of rows to generate (up to 10,000).
- Specify the output file path (or use the default path in the outputs directory).
- Select the OpenAI model, temperature, and batch size for data generation.

### Command-Line Arguments
You can also run the script with arguments:

python script_name.py --excel path/to/schema.xlsx --rows 200 --output my_data.xlsx --model gpt-4o-mini --temperature 0.7 --batch-size 25

Alternatively, use:
- `--columns` to provide a comma-separated list of column names.
- `--prompt` to describe the data you want to generate in natural language.
At least one of `--excel`, `--columns`, or `--prompt` must be provided.

## Key Functionalities

- **Schema Generation:**
  - Generates a data schema from a natural language prompt using OpenAI's API.
  - Alternatively, reads schema from an Excel file or manual input.
- **Data Generation:**
  - Generates synthetic data in batches using parallel processing with ThreadPoolExecutor.
  - Maintains consistency by including example rows and previously generated data in prompts.
- **Data Formatting:**
  - Analyzes example data to determine formats (e.g., date formats, numeric precision, categorical values) and includes these instructions in the prompt.
- **Output:**
  - Saves the synthetic data as an Excel file in the outputs directory.
  - Creates a detailed log file with timestamps and progress information.
- **Error Handling & Logging:**
  - Logs all major steps and errors to a log file stored in the outputs directory.

## Troubleshooting

- **Missing API Key:**  
  Ensure your `.env` file contains a valid `OPENAI_API_KEY`.

- **Invalid Input:**  
  Verify that the Excel file path is correct and that you provide valid input when prompted.

- **Data Generation Issues:**  
  Check the log file in the outputs directory for detailed error messages if the data generation fails.

- **Output File Not Saved:**  
  Ensure you have write permissions to the output directory.

## Conclusion
This Synthetic Data Generator Tool leverages OpenAI's API to create realistic synthetic datasets based on user-defined schemas. Whether you provide an Excel file, manually enter column names, or describe the data in plain language, the tool generates data in batches with consistent formatting. It is ideal for testing, training models, or any scenario where synthetic data is needed. Customize the parameters to suit your needs and explore the flexibility of synthetic data generation with this powerful script!

