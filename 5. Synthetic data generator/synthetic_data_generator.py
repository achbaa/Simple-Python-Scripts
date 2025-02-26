import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
import datetime

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create Output files directory if it doesn't exist (renamed from "outputs")
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "Output files")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Configure logging to the Output files directory
log_file = os.path.join(OUTPUTS_DIR, f"synthetic_data_generation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logging.info(f"Script directory: {SCRIPT_DIR}")
logging.info(f"Outputs directory: {OUTPUTS_DIR}")
logging.info(f"Log file: {log_file}")

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("Error: OPENAI_API_KEY not found in environment variables.")
    print("Please create a .env file with your OpenAI API key or set it as an environment variable.")
    sys.exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_schema_from_prompt(prompt, model="gpt-4o"):
    """
    Generate column names and example data based on a user prompt.
    
    Args:
        prompt (str): User's description of the data they want to generate.
        model (str): OpenAI model to use for generating the schema.
        
    Returns:
        tuple: (column_names, example_rows)
    """
    logging.info("Generating schema from user prompt...")
    
    system_prompt = """
    You are an expert data scientist tasked with creating synthetic data schemas.
    Based on the user's description, create a suitable data schema with column names and 3 example rows.
    The output should be in JSON format with two keys:
    1. "columns": a list of column names
    2. "examples": a list of lists, where each inner list is an example row
    
    Ensure all column names are appropriate for Excel and follow best practices.
    The example data should be realistic and follow consistent formats.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create a data schema for: {prompt}"}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        try:
            data = json.loads(content)
            column_names = data.get("columns", [])
            example_rows = data.get("examples", [])
            
            # Validate the schema
            if not column_names:
                logging.error("Generated schema contains no column names")
                return None, None
                
            if not example_rows:
                logging.warning("Generated schema contains no example rows")
                
            # Print generated schema
            print("\nGenerated Schema:")
            print(f"Columns: {column_names}")
            print("Example data:")
            for row in example_rows:
                print(f"  {row}")
                
            return column_names, example_rows
                
        except json.JSONDecodeError:
            logging.error(f"Failed to parse schema JSON: {content[:200]}...")
            return None, None
            
    except Exception as e:
        logging.error(f"Error generating schema: {str(e)}")
        return None, None

class SyntheticDataGenerator:
    """Class for generating synthetic data using OpenAI's API."""
    
    def __init__(self, column_names, example_rows=None, model="gpt-4o-mini", 
                 temperature=0.7, batch_size=25, max_workers=4, max_retries=3):
        """
        Initialize the synthetic data generator.
        
        Args:
            column_names (list): List of column names for the synthetic data.
            example_rows (list, optional): List of example rows to guide data generation.
            model (str, optional): OpenAI model to use. Defaults to "gpt-4o-mini".
            temperature (float, optional): Controls randomness in generation. Defaults to 0.7.
            batch_size (int, optional): Number of rows to generate per API call. Defaults to 25.
            max_workers (int, optional): Max number of parallel workers. Defaults to 4.
            max_retries (int, optional): Max number of retries for API calls. Defaults to 3.
        """
        self.column_names = column_names
        self.example_rows = example_rows if example_rows else []
        self.model = model
        self.temperature = temperature
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.data_format_instructions = ""
        
        # Determine data types and format from example rows
        self._analyze_example_data()
        
    def _analyze_example_data(self):
        """Analyze example data to determine data types and formats."""
        if not self.example_rows:
            return
        
        # Build format instructions based on example data
        format_instructions = []
        example_df = pd.DataFrame(self.example_rows, columns=self.column_names)
        
        for col in self.column_names:
            sample_values = example_df[col].dropna().tolist()
            if not sample_values:
                continue
                
            # Detect if column might be dates
            date_formats = []
            try:
                if pd.to_datetime(sample_values, errors='coerce').notna().all():
                    # Get the date format
                    for value in sample_values:
                        if '-' in str(value) and len(str(value)) == 10:  # YYYY-MM-DD
                            date_formats.append("YYYY-MM-DD")
                        elif '/' in str(value) and len(str(value)) == 10:  # MM/DD/YYYY or DD/MM/YYYY
                            if int(str(value).split('/')[0]) <= 12:
                                date_formats.append("MM/DD/YYYY or DD/MM/YYYY")
                            else:
                                date_formats.append("DD/MM/YYYY")
                    
                    if date_formats:
                        common_format = max(set(date_formats), key=date_formats.count)
                        format_instructions.append(f"Column '{col}' should be formatted as dates in {common_format} format.")
            except:
                pass
                
            # Check for numeric columns
            try:
                numeric_values = pd.to_numeric(sample_values, errors='coerce')
                if numeric_values.notna().all():
                    if all(float(x).is_integer() for x in numeric_values):
                        format_instructions.append(f"Column '{col}' should contain integer values.")
                    else:
                        # Check decimal precision
                        precisions = [len(str(x).split('.')[-1]) for x in sample_values if '.' in str(x)]
                        if precisions:
                            avg_precision = sum(precisions) / len(precisions)
                            format_instructions.append(f"Column '{col}' should contain decimal values with approximately {int(avg_precision)} decimal places.")
                        else:
                            format_instructions.append(f"Column '{col}' should contain numeric values.")
            except:
                pass
                
            # Check for categoricals
            if len(set(sample_values)) < len(sample_values) * 0.5 and len(set(sample_values)) < 10:
                categories = list(set(sample_values))
                format_instructions.append(f"Column '{col}' should contain categorical values, likely one of: {', '.join(str(c) for c in categories)}.")
                
        self.data_format_instructions = "\n".join(format_instructions)
        
    def _create_prompt(self, batch_size, existing_data=None):
        """
        Create a prompt for the OpenAI API to generate synthetic data.
        
        Args:
            batch_size (int): Number of rows to generate.
            existing_data (list, optional): Previously generated data to maintain consistency.
        
        Returns:
            str: The prompt for the OpenAI API.
        """
        prompt = f"Generate {batch_size} rows of synthetic data with the following columns:\n"
        prompt += ", ".join(self.column_names)
        
        # Add example rows
        if self.example_rows:
            prompt += "\n\nHere are some example rows to follow:\n"
            for row in self.example_rows:
                prompt += str(row) + "\n"
                
        # Add previously generated data to maintain consistency
        if existing_data and len(existing_data) > 0:
            # Fix: Make sure we're sampling from a 1D array of row indices
            sample_size = min(5, len(existing_data))
            sample_indices = np.random.choice(range(len(existing_data)), sample_size, replace=False)
            samples = [existing_data[i] for i in sample_indices]
            
            prompt += f"\n\nHere are {sample_size} samples from previously generated data (maintain consistency with these):\n"
            for row in samples:
                prompt += str(row) + "\n"
                
        # Add format instructions
        if self.data_format_instructions:
            prompt += f"\n\nPlease follow these format instructions:\n{self.data_format_instructions}\n"
            
        prompt += "\nEnsure the data is realistic and consistent. Return the data as a JSON array of arrays, where each inner array represents a row."
        
        return prompt
        
    def _generate_batch(self, batch_size, existing_data=None):
        """
        Generate a batch of synthetic data rows.
        
        Args:
            batch_size (int): Number of rows to generate.
            existing_data (list, optional): Previously generated data for consistency.
            
        Returns:
            list: The generated rows of synthetic data.
        """
        prompt = self._create_prompt(batch_size, existing_data)
        
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a synthetic data generator that creates realistic sample data based on the provided columns and examples. Always return data in the exact format requested."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                
                try:
                    # Parse the JSON response
                    data = json.loads(content)
                    
                    # Handle different possible JSON structures
                    if isinstance(data, list):
                        return data
                    elif "data" in data:
                        return data["data"]
                    elif "rows" in data:
                        return data["rows"]
                    else:
                        # Try to find any array in the response
                        for key, value in data.items():
                            if isinstance(value, list) and len(value) > 0:
                                return value
                                
                        logging.warning(f"Unexpected JSON structure: {content[:200]}...")
                        continue
                        
                except json.JSONDecodeError:
                    logging.warning(f"Failed to parse JSON on attempt {attempt+1}: {content[:200]}...")
                    if attempt < self.max_retries - 1:
                        time.sleep(2)
                    continue
                    
            except Exception as e:
                logging.warning(f"API call failed on attempt {attempt+1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                continue
                
        # If all attempts failed, return empty list
        logging.error(f"All {self.max_retries} attempts to generate batch failed.")
        return []
        
    def generate_data(self, num_rows):
        """
        Generate synthetic data.
        
        Args:
            num_rows (int): Number of rows to generate.
            
        Returns:
            pandas.DataFrame: The generated synthetic data.
        """
        logging.info(f"Generating {num_rows} rows of synthetic data...")
        
        # Calculate number of batches
        num_batches = (num_rows + self.batch_size - 1) // self.batch_size
        
        # Generate in batches
        all_rows = []
        
        # Generate first batch to serve as examples for later batches
        if num_batches > 0:
            first_batch_size = min(self.batch_size, num_rows)
            first_batch = self._generate_batch(first_batch_size)
            all_rows.extend(first_batch)
            logging.info(f"Generated first batch: {len(first_batch)} rows")
            
        # Generate remaining batches in parallel
        if num_batches > 1:
            remaining_rows = num_rows - len(all_rows)
            remaining_batches = (remaining_rows + self.batch_size - 1) // self.batch_size
            
            batch_sizes = [self.batch_size] * remaining_batches
            # Adjust last batch size if needed
            if remaining_rows % self.batch_size != 0:
                batch_sizes[-1] = remaining_rows % self.batch_size
                
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create a copy of existing data for each worker
                existing_data = all_rows.copy()
                
                # Submit batch generation tasks
                future_to_batch = {
                    executor.submit(self._generate_batch, size, existing_data): i 
                    for i, size in enumerate(batch_sizes)
                }
                
                # Process completed batches
                for future in tqdm(future_to_batch, desc="Generating batches", total=len(batch_sizes)):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_rows = future.result()
                        all_rows.extend(batch_rows)
                        logging.info(f"Generated batch {batch_idx+1}/{len(batch_sizes)}: {len(batch_rows)} rows")
                    except Exception as e:
                        logging.error(f"Error generating batch {batch_idx+1}: {str(e)}")
        
        # Ensure we have the right number of rows
        if len(all_rows) > num_rows:
            all_rows = all_rows[:num_rows]
            
        # Convert to DataFrame
        if all_rows:
            # Ensure each row has the correct number of elements
            valid_rows = [row for row in all_rows if len(row) == len(self.column_names)]
            if len(valid_rows) < len(all_rows):
                logging.warning(f"Filtered out {len(all_rows) - len(valid_rows)} rows with incorrect number of elements.")
                
            if valid_rows:
                df = pd.DataFrame(valid_rows, columns=self.column_names)
                return df
            else:
                logging.error("No valid rows generated.")
                return pd.DataFrame(columns=self.column_names)
        else:
            logging.error("Failed to generate any data.")
            return pd.DataFrame(columns=self.column_names)

def read_excel_structure(file_path):
    """
    Read column names and example rows from an Excel file.
    
    Args:
        file_path (str): Path to the Excel file.
        
    Returns:
        tuple: (column_names, example_rows)
    """
    try:
        df = pd.read_excel(file_path)
        column_names = df.columns.tolist()
        
        # Get example rows
        num_examples = min(5, len(df))
        example_rows = df.head(num_examples).values.tolist()
        
        return column_names, example_rows
    except Exception as e:
        logging.error(f"Error reading Excel file: {str(e)}")
        return None, None

def get_user_input():
    """
    Get input from the user interactively.
    
    Returns:
        dict: User input parameters.
    """
    print("\n=== Synthetic Data Generator ===\n")
    
    # Ask for input method
    print("How would you like to specify the data structure?")
    print("1. Provide an Excel file with column names and example rows")
    print("2. Enter column names manually")
    print("3. Describe the data you want in natural language")
    
    while True:
        input_method = input("Select an option (1-3): ").strip()
        if input_method in ["1", "2", "3"]:
            break
        print("Please enter 1, 2, or 3.")
        
    column_names = []
    example_rows = []
    
    if input_method == "1":
        # Get Excel file path
        while True:
            excel_path = input("Enter the path to the Excel file: ").strip()
            if os.path.isfile(excel_path) and excel_path.endswith(('.xlsx', '.xls')):
                column_names, example_rows = read_excel_structure(excel_path)
                if column_names:
                    break
            print("Please enter a valid Excel file path.")
    elif input_method == "2":
        # Get column names manually
        print("\nEnter column names (one per line, press Enter twice to finish):")
        while True:
            column = input().strip()
            if not column:
                break
            column_names.append(column)
            
        # Ask if user wants to provide example values
        provide_examples = input("\nWould you like to provide example values? (y/n, default is n): ").strip().lower()
        if provide_examples == "y":
            num_examples = 1
            while num_examples <= 5:
                print(f"\nEnter values for example row {num_examples} (one per column):")
                row = []
                for col in column_names:
                    value = input(f"{col}: ").strip()
                    row.append(value)
                example_rows.append(row)
                
                if num_examples < 5:
                    more = input("Add another example row? (y/n, default is n): ").strip().lower()
                    if more != "y":
                        break
                num_examples += 1
    else:  # input_method == "3"
        # Get data description from user
        print("\nDescribe the data you want to generate (be as specific as possible):")
        data_description = input().strip()
        
        while not data_description:
            print("Please provide a description:")
            data_description = input().strip()
        
        # Get model for schema generation
        schema_model = "gpt-4o"  # Default to the most capable model for schema generation
        
        # Generate schema
        column_names, example_rows = generate_schema_from_prompt(data_description, schema_model)
        
        if not column_names:
            print("Failed to generate schema from description. Please try again with a more detailed description.")
            sys.exit(1)
        
        # Allow user to modify the schema
        modify = input("\nWould you like to modify the generated schema? (y/n, default is n): ").strip().lower()
        if modify == "y":
            # Show current columns
            print("\nCurrent columns:")
            for i, col in enumerate(column_names, 1):
                print(f"{i}. {col}")
                
            # Allow adding, removing, or renaming
            print("\nOptions:")
            print("1. Add column")
            print("2. Remove column")
            print("3. Rename column")
            print("4. Continue with current schema")
            
            while True:
                choice = input("Select an option (1-4): ").strip()
                
                if choice == "1":  # Add column
                    new_col = input("Enter new column name: ").strip()
                    if new_col:
                        column_names.append(new_col)
                        # Add empty values to example rows
                        for row in example_rows:
                            row.append("")
                        print(f"Added column: {new_col}")
                elif choice == "2":  # Remove column
                    idx = int(input("Enter column number to remove: ").strip()) - 1
                    if 0 <= idx < len(column_names):
                        removed = column_names.pop(idx)
                        # Remove values from example rows
                        for row in example_rows:
                            if idx < len(row):
                                row.pop(idx)
                        print(f"Removed column: {removed}")
                elif choice == "3":  # Rename column
                    idx = int(input("Enter column number to rename: ").strip()) - 1
                    if 0 <= idx < len(column_names):
                        new_name = input(f"Enter new name for '{column_names[idx]}': ").strip()
                        if new_name:
                            column_names[idx] = new_name
                            print(f"Renamed column to: {new_name}")
                elif choice == "4":  # Continue
                    break
                
                print("\nUpdated columns:")
                for i, col in enumerate(column_names, 1):
                    print(f"{i}. {col}")
                
                another = input("\nMake another change? (y/n, default is n): ").strip().lower()
                if another != "y":
                    break
                
    # Get number of rows to generate
    while True:
        try:
            num_rows = int(input("\nEnter number of rows to generate (max 10,000): ").strip())
            if 1 <= num_rows <= 10000:
                break
            print("Please enter a number between 1 and 10,000.")
        except ValueError:
            print("Please enter a valid number.")
            
    # Get output file path
    output_path = input("\nEnter the path for the output Excel file (leave blank for default): ").strip()
    if not output_path:
        # Default to Output files directory with a timestamp in the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUTS_DIR, f"synthetic_data_{timestamp}.xlsx")
    elif not os.path.isabs(output_path):
        # If a relative path is provided, make it relative to the Output files directory
        if not output_path.endswith(('.xlsx')):
            output_path += ".xlsx"
        output_path = os.path.join(OUTPUTS_DIR, output_path)
    elif not output_path.endswith(('.xlsx')):
        output_path += ".xlsx"
        
    # Get model
    models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
        
    while True:
        model_choice = input(f"Select a model (1-{len(models)}, default is 2): ").strip()
        if not model_choice:
            model = models[1]  # Default to gpt-4o-mini
            break
        try:
            model_index = int(model_choice) - 1
            if 0 <= model_index < len(models):
                model = models[model_index]
                break
            else:
                print(f"Please enter a number between 1 and {len(models)}.")
        except ValueError:
            print("Please enter a valid number.")
            
    # Get temperature
    while True:
        try:
            temp = input("\nEnter temperature value (0.0-1.0, default is 0.7): ").strip()
            if not temp:
                temperature = 0.7
                break
            temperature = float(temp)
            if 0.0 <= temperature <= 1.0:
                break
            print("Please enter a number between 0.0 and 1.0.")
        except ValueError:
            print("Please enter a valid number.")
            
    # Get batch size
    while True:
        try:
            batch = input("\nEnter batch size (5-50, default is 25): ").strip()
            if not batch:
                batch_size = 25
                break
            batch_size = int(batch)
            if 5 <= batch_size <= 50:
                break
            print("Please enter a number between 5 and 50.")
        except ValueError:
            print("Please enter a valid number.")
            
    return {
        "column_names": column_names,
        "example_rows": example_rows,
        "num_rows": num_rows,
        "output_path": output_path,
        "model": model,
        "temperature": temperature,
        "batch_size": batch_size
    }

def main():
    """Main function to run the script."""
    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        # Use argparse for command-line arguments
        parser = argparse.ArgumentParser(description='Generate synthetic data using OpenAI API.')
        parser.add_argument('--excel', help='Path to Excel file with column definitions and examples')
        parser.add_argument('--columns', help='Comma-separated list of column names')
        parser.add_argument('--prompt', help='Natural language description of the data to generate')
        parser.add_argument('--rows', type=int, default=100, help='Number of rows to generate (default: 100)')
        parser.add_argument('--output', default=None, help='Output Excel file path (default is outputs/synthetic_data_TIMESTAMP.xlsx)')
        parser.add_argument('--model', default='gpt-4o-mini', help='OpenAI model to use')
        parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation (0.0-1.0)')
        parser.add_argument('--batch-size', type=int, default=25, help='Batch size for generation')
        
        args = parser.parse_args()
        
        column_names = []
        example_rows = []
        
        if args.prompt:
            # Generate schema from prompt
            column_names, example_rows = generate_schema_from_prompt(args.prompt)
            if not column_names:
                print("Error: Failed to generate schema from prompt.")
                sys.exit(1)
        elif args.excel and os.path.isfile(args.excel):
            column_names, example_rows = read_excel_structure(args.excel)
        elif args.columns:
            column_names = [col.strip() for col in args.columns.split(',')]
            example_rows = []
        else:
            print("Error: Either --excel, --columns, or --prompt must be provided.")
            sys.exit(1)
            
        num_rows = max(1, min(args.rows, 10000))
        
        # Handle output path
        if args.output:
            if os.path.isabs(args.output):
                output_path = args.output
            else:
                output_path = os.path.join(OUTPUTS_DIR, args.output)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUTS_DIR, f"synthetic_data_{timestamp}.xlsx")
            
        if not output_path.endswith('.xlsx'):
            output_path += '.xlsx'
        
        model = args.model
        temperature = args.temperature
        batch_size = args.batch_size
    else:
        # Get input interactively
        user_input = get_user_input()
        column_names = user_input["column_names"]
        example_rows = user_input["example_rows"]
        num_rows = user_input["num_rows"]
        output_path = user_input["output_path"]
        model = user_input["model"]
        temperature = user_input["temperature"]
        batch_size = user_input["batch_size"]
    
    # Initialize generator
    generator = SyntheticDataGenerator(
        column_names=column_names,
        example_rows=example_rows,
        model=model,
        temperature=temperature,
        batch_size=batch_size
    )
    
    # Generate data
    start_time = time.time()
    data = generator.generate_data(num_rows)
    end_time = time.time()
    
    # Save to Excel
    if not data.empty:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        data.to_excel(output_path, index=False)
        print(f"\nGenerated {len(data)} rows of synthetic data in {end_time - start_time:.2f} seconds.")
        print(f"Data saved to {output_path}")
        
        # Also save a copy of the log file with the same base name
        log_output = os.path.splitext(output_path)[0] + "_log.txt"
        try:
            # Copy the current log file to the same location as the output file
            with open(log_file, 'r') as src, open(log_output, 'w') as dst:
                dst.write(src.read())
            print(f"Log saved to {log_output}")
        except Exception as e:
            logging.error(f"Error saving log file: {str(e)}")
    else:
        print("\nFailed to generate data. Check the log file for details.")
        print(f"Log file location: {log_file}")

if __name__ == "__main__":
    main()
