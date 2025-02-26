# Python AI Tools Collection

## Overview

This repository contains a collection of practical Python tools that leverage the OpenAI API to automate and enhance various data processing tasks. These tools are designed specifically for users who may not have extensive technical backgrounds, providing immediate business value across a range of use cases.

## Who Is This For?

This repository is tailored for:
- Users with limited or no Python experience
- Professionals looking to automate routine data processing tasks
- Teams exploring how to integrate AI into their deliverables
- Anyone interested in learning how AI can transform standard workflows

## What's Included

This collection contains 6 powerful tools, each in its own folder with detailed documentation:

1. **PDF Summaries** - Automatically extract and summarize key content from PDF documents, perfect for research, report analysis, and document processing.

2. **Excel Categorization** - Intelligently categorize data in spreadsheets, helping with product classification, customer segmentation, and data organization.

3. **Interview Transcript Synthesizer** - Transform interview transcripts into structured insights, themes, and actionable summaries.

4. **Interview Audio Transcriber** - Convert interview audio recordings to text with options for summarization, making qualitative research more efficient.

5. **Synthetic Data Generator** - Create realistic synthetic datasets for testing, training, or demonstration purposes.

6. **Flexible Data Manipulator** - Apply custom transformations to spreadsheet data using natural language instructions.

## Getting Started

### Step 1: Set Up Your Environment
1. **Install Git** if you haven't already:
   - Download from [git-scm.com](https://git-scm.com/downloads)
   - Follow the installation instructions for your operating system

2. **Clone this Repository**:
   ```
   git clone [repository-url]
   ```
   Replace `[repository-url]` with the actual URL of this repository.

3. **Install Visual Studio Code**:
   - Download from [code.visualstudio.com](https://code.visualstudio.com/)
   - Follow the installation instructions for your operating system

### Step 2: Get Your OpenAI API Key
1. Sign up for an OpenAI account at [platform.openai.com](https://platform.openai.com/)
2. Navigate to the API section and create a new API key
3. Create a file named `.env` in the root directory of this repository
4. Add your API key to the `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
   Replace `your_api_key_here` with your actual API key.

### Step 3: Open the Project in VS Code
1. Launch VS Code
2. Go to File → Open Folder
3. Navigate to the cloned repository folder and click "Open"

### Step 4: Set Up Python Environment
1. Install Python if you haven't already:
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH" (important!)

2. Open the Terminal in VS Code:
   - Go to: Terminal → New Terminal
   - You'll see a command prompt appear at the bottom of VS Code

3. Create a virtual environment (keeps your project dependencies organized):
   ```
   # Windows
   python -m venv venv

   # Mac/Linux
   python3 -m venv venv
   ```

4. Activate the virtual environment:
   ```
   # Windows
   venv\Scripts\activate

   # Mac/Linux
   source venv/bin/activate
   ```
   You'll see `(venv)` appear at the beginning of your command prompt.

5. Install the required packages:
   
   **Option A: Install all dependencies at once** (recommended for beginners):
   ```
   pip install -r requirements.txt
   ```
   
   **Option B: Install only dependencies for a specific tool**:
   - Navigate to the specific tool folder (replace `tool-folder` with actual folder name):
     ```
     cd "tool-folder"
     ```
   - Install dependencies for just that tool:
     ```
     pip install -r requirements.txt
     ```
   - Wait for installation to complete (you'll see progress indicators)

## Recommended Learning Path

If you're new to these tools, we recommend this learning sequence:

1. Start with **Excel Categorization** - It's the simplest tool to understand and use
2. Move to **PDF Summaries** - Learn how AI can process unstructured text from documents
3. Try **Synthetic Data Generator** - Understand how to create realistic data for data analysis or AI testing
4. Explore the other tools as needed for your specific work

## Customization for Your Work

You're encouraged to modify these tools for your specific needs:
- Key parameters are accessible through interactive prompts (no coding required)
- Feel free to experiment and adapt the functionality to meet your requirements
- Share your modifications with colleagues to build collective capabilities

## Support for Users

- Each subfolder contains detailed troubleshooting information
- Reach out to the repository maintainer if you have any questions or suggestions
- For troubleshooting and learning, utilize ChatGPT to get real-time assistance and guidance
- Check OpenAI's documentation for API-specific questions: https://platform.openai.com/docs/

---

Explore each folder to find detailed READMEs with installation instructions, usage examples, and customization options.

Happy coding and automating your work!
