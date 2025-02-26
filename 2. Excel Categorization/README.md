# Excel Data Categorization Tool

## Overview
This Python script categorizes data from an Excel file using OpenAI's API. It reads data from a specified source column in an Excel file, processes the data in batches, and writes the categorized results to a target column. The script supports both interactive mode (with user prompts) and quick mode (using command-line arguments), making it ideal for beginners who want to automate data categorization tasks.

## Prerequisites
- **Python 3:** Ensure Python 3 is installed on your system.
- **Visual Studio Code (VS Code):** A beginner-friendly code editor.
- **OpenAI API Key:** Required for accessing OpenAI's categorization capabilities.
- **Required Python Libraries:** 
  - openpyxl
  - pandas
  - python-dotenv
  - openai
  - argparse (part of the standard library)
  - re (part of the standard library)
- **Basic Command Line Skills:** Familiarity with running scripts in a terminal.

## Installation

1. **Download or Clone the Repository:**
   - If you're new to GitHub, click the "Code" button on the repository page and select "Download ZIP". Extract the ZIP file to a folder on your computer.

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
   - If a `requirements.txt` file is not provided, install the libraries individually using pip.

## Setup

1. **Configure Environment Variables:**
   - Create a file named `.env` in the project folder.
   - Add your OpenAI API key in the following format:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - Replace `your_api_key_here` with your actual API key.

## How to Run the Script

1. Open the integrated terminal in VS Code.
2. Run the script by typing:

python script_name.py

Replace `script_name.py` with the actual name of the Python file.

3. **Interactive Mode:**
- The script will prompt you for:
  - The path to your Excel file.
  - Source column number (the column with data to categorize).
  - Target column number (where the categories will be written).
  - Starting row number (after the header row).
  - Batch size (number of rows to process at a time).
  - Optionally, allowed categories and custom prompts.
- A preview of the first few rows will be displayed to help you identify the correct columns.

4. **Quick Mode:**
- You can bypass the interactive prompts by providing command-line arguments:
  ```
  python script_name.py --quick --file your_excel_file.xlsx --source 1 --target 2 --start 2 --batch 5 --model gpt-4o-mini --categories "Category1,Category2"
  ```

## Key Functionalities

- **Excel Data Processing:** Reads data from an Excel file using openpyxl and writes the categorized results back to the file.
- **Batch Categorization:** Processes data in batches for efficiency, reducing the number of API calls.
- **OpenAI Integration:** Uses the OpenAI API to categorize each item based on provided prompts.
- **Custom Prompts and Allowed Categories:** Offers the ability to specify custom system prompts and limit responses to a predefined set of allowed categories.
- **Interactive and Quick Modes:** Supports both interactive prompts and quick command-line options to cater to different user preferences.
- **Progress and Error Handling:** Displays progress updates and handles errors gracefully, including saving progress periodically.

## Troubleshooting

- **Missing API Key:** Ensure your `.env` file contains a valid `OPENAI_API_KEY`.
- **File Not Found:** Verify that the Excel file path is correct.
- **Invalid Column or Row Numbers:** Use the preview functionality to confirm the correct column and row numbers.
- **API Errors:** Check your internet connection and ensure your OpenAI API key is valid.

## Conclusion
This Excel Data Categorization Tool is designed to help beginners automate the process of categorizing data in Excel using advanced AI. Its flexibility through both interactive and quick modes makes it a versatile tool for various data categorization tasks. Enjoy customizing and extending the tool to fit your specific needs!

