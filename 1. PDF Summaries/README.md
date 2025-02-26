# PDF Summaries Tool

## Overview
This tool automatically extracts and summarizes content from PDF documents using OpenAI's API. It's perfect for quickly digesting research papers, reports, presentations, and other PDF documents.

## Features
- Extract text from single or multiple PDF files
- Generate concise summaries with key points
- Process large documents by breaking them into manageable chunks
- Save summaries as text files for easy reference

## Prerequisites
- Python 3.8 or higher
- OpenAI API key from BCG's OpenAI Portal
- Required Python packages (install using `pip install -r requirements.txt`):
  - python-dotenv
  - openai
  - PyPDF2
  - backoff

## Quick Start Guide

### 1. Set Up Your Environment
Create a `.env` file in this directory with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-from-bcg-openai-portal
```

### 2. Run the Tool
From the command line:
```
python pdf_summarizer.py
```

Or with command-line arguments:
```
python pdf_summarizer.py --file "path/to/document.pdf" --model "gpt-4o" --output "summary.txt"
```

### 3. Follow the Interactive Prompts
If running without command-line arguments, you'll be prompted to:
- Enter the path to your PDF file(s)
- Select the OpenAI model to use
- Specify output preferences

### 4. View Your Results
The tool will create a summary file in the outputs directory with:
- Document metadata
- Executive summary
- Key points
- Main themes

## Example Usage

### Example 1: Summarize a Research Paper
```
python pdf_summarizer.py --file "examples/research_paper.pdf" --model "gpt-4o"
```

### Example 2: Process Multiple Documents
```
python pdf_summarizer.py --directory "examples/reports" --model "gpt-4o-mini"
```

## Troubleshooting

### Common Issues
- **"PDF file not found"**: Verify the file path is correct and the file exists
- **"API key not found"**: Ensure your `.env` file exists and contains a valid API key
- **"Token limit exceeded"**: Try using a model with a larger context window or adjust chunk size

### Getting Help
If you encounter issues not covered here, please reach out to Alexander Achba or someone from BCG X.

## Customization
You can modify the summarization prompts in the code to focus on specific aspects of documents relevant to your case work.
