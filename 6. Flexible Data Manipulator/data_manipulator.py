import os
import sys
import pandas as pd
import numpy as np
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import logging
import datetime
import argparse
import json
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create Output files directory if it doesn't exist (renamed from "outputs")
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "Output files")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Configure logging to the Output files directory
log_file = os.path.join(OUTPUTS_DIR, f"data_manipulation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

class DataManipulator:
    """Class for manipulating data using OpenAI's API."""
    
    def __init__(self, df, input_columns, output_columns, transformation_prompt, 
                 model="gpt-3.5-turbo", temperature=0.3, batch_size=20, max_workers=4, max_retries=3):
        """
        Initialize the data manipulator.
        
        Args:
            df (pandas.DataFrame): Input dataframe to manipulate.
            input_columns (list): List of column names to use as input.
            output_columns (list): List of column names to create as output.
            transformation_prompt (str): Prompt describing how to transform the input data.
            model (str, optional): OpenAI model to use. Defaults to "gpt-3.5-turbo".
            temperature (float, optional): Controls randomness in generation. Defaults to 0.3.
            batch_size (int, optional): Number of rows to process per API call. Defaults to 20.
            max_workers (int, optional): Max number of parallel workers. Defaults to 4.
            max_retries (int, optional): Max number of retries for API calls. Defaults to 3.
        """
        self.df = df
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.transformation_prompt = transformation_prompt
        self.model = model
        self.temperature = temperature
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        
        # Create output columns if they don't exist
        for col in output_columns:
            if col not in df.columns:
                self.df[col] = ""
                
        # Get sample data for examples
        self._get_sample_data()
                
    def _get_sample_data(self, num_samples=5):
        """
        Get sample data for examples in the prompt.
        
        Args:
            num_samples (int, optional): Number of sample rows to use. Defaults to 5.
        """
        if len(self.df) <= num_samples:
            self.sample_indices = list(range(len(self.df)))
        else:
            self.sample_indices = np.random.choice(len(self.df), num_samples, replace=False)
            
        self.samples = self.df.iloc[self.sample_indices]
        
    def _create_system_prompt(self):
        """
        Create the system prompt for the OpenAI API.
        
        Returns:
            str: The system prompt.
        """
        system_prompt = """
        You are a data transformation assistant. Your task is to transform input data according to specific instructions.
        Always return your answers in valid JSON format with a "results" key that contains an array of transformed values.
        For each input row, you should generate exactly one corresponding output value for each requested output column.
        
        For example, if processing 3 rows with 2 output columns, your response should be:
        {
          "results": [
            ["output1_row1", "output2_row1"],
            ["output1_row2", "output2_row2"],
            ["output1_row3", "output2_row3"]
          ]
        }
        
        Be consistent in your transformations and follow the instructions carefully.
        """
        return system_prompt.strip()
    
    def _create_batch_prompt(self, batch_indices):
        """
        Create a prompt for a batch of data rows.
        
        Args:
            batch_indices (list): List of row indices to include in the batch.
            
        Returns:
            str: The prompt for the batch.
        """
        batch_data = self.df.iloc[batch_indices]
        
        prompt = f"Transform the following data according to these instructions:\n\n{self.transformation_prompt}\n\n"
        
        # Add information about input and output columns
        prompt += f"Input columns: {', '.join(self.input_columns)}\n"
        prompt += f"Output columns: {', '.join(self.output_columns)}\n\n"
        
        # Add examples if we have them and this isn't the first batch
        if hasattr(self, 'samples') and not all(idx in self.sample_indices for idx in batch_indices):
            prompt += "Here are some examples of how the data should be transformed:\n\n"
            for _, row in self.samples.iterrows():
                input_data = {col: row[col] for col in self.input_columns}
                output_data = {col: row[col] if col in row and pd.notna(row[col]) else "" for col in self.output_columns}
                prompt += f"Input: {json.dumps(input_data)}\n"
                prompt += f"Output: {json.dumps(output_data)}\n\n"
        
        # Add the batch data to transform
        prompt += "Now, transform the following data:\n\n"
        for i, (_, row) in enumerate(batch_data.iterrows()):
            input_data = {col: str(row[col]) if pd.notna(row[col]) else "" for col in self.input_columns}
            prompt += f"Row {i+1}: {json.dumps(input_data)}\n"
            
        prompt += "\nReturn the transformed data as a JSON object with a 'results' key containing an array of arrays."
        prompt += "\nEach inner array should contain the transformed values for one row, in the order of the output columns."
        
        return prompt
    
    def _process_batch(self, batch_indices):
        """
        Process a batch of data rows.
        
        Args:
            batch_indices (list): List of row indices to process.
            
        Returns:
            list: The transformed data for each row.
        """
        prompt = self._create_batch_prompt(batch_indices)
        
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._create_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                
                try:
                    # Parse the JSON response
                    data = json.loads(content)
                    
                    if "results" not in data:
                        logging.warning(f"Missing 'results' key in response: {content[:200]}...")
                        if attempt < self.max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        else:
                            return [["Error: Invalid response format"]] * len(batch_indices)
                    
                    results = data["results"]
                    
                    # Validate results
                    if len(results) != len(batch_indices):
                        logging.warning(f"Expected {len(batch_indices)} results, got {len(results)}")
                        # Try to recover by padding or truncating
                        if len(results) < len(batch_indices):
                            # Pad with empty results
                            results.extend([[""] * len(self.output_columns)] * (len(batch_indices) - len(results)))
                        else:
                            # Truncate
                            results = results[:len(batch_indices)]
                    
                    # Ensure each result has the correct number of output values
                    for i, result in enumerate(results):
                        if not isinstance(result, list):
                            logging.warning(f"Result {i} is not a list: {result}")
                            results[i] = [str(result)] * len(self.output_columns)
                        elif len(result) != len(self.output_columns):
                            logging.warning(f"Result {i} has wrong number of values: {result}")
                            if len(result) < len(self.output_columns):
                                # Pad with empty strings
                                results[i].extend([""] * (len(self.output_columns) - len(result)))
                            else:
                                # Truncate
                                results[i] = result[:len(self.output_columns)]
                    
                    return results
                    
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse JSON: {content[:200]}...")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return [["Error: Invalid JSON response"]] * len(batch_indices)
            
            except Exception as e:
                logging.error(f"Error calling API: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return [["Error: " + str(e)]] * len(batch_indices)
        
        # If we get here, all retries failed
        return [["Error: All retries failed"]] * len(batch_indices)
    
    def process_data(self):
        """
        Process all data in the dataframe.
        
        Returns:
            pandas.DataFrame: The processed dataframe.
        """
        num_rows = len(self.df)
        
        logging.info(f"Processing {num_rows} rows...")
        
        # Create batches
        all_indices = list(range(num_rows))
        batch_count = (num_rows + self.batch_size - 1) // self.batch_size
        batches = []
        
        for i in range(batch_count):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_rows)
            batches.append(all_indices[start_idx:end_idx])
        
        results = []
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {executor.submit(self._process_batch, batch): i for i, batch in enumerate(batches)}
            
            with tqdm(total=len(batches), desc="Processing batches") as pbar:
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    batch_indices = batches[batch_idx]
                    
                    try:
                        batch_results = future.result()
                        
                        # Store results by row index
                        for i, row_idx in enumerate(batch_indices):
                            if i < len(batch_results):
                                row_result = batch_results[i]
                                for j, col in enumerate(self.output_columns):
                                    if j < len(row_result):
                                        self.df.at[row_idx, col] = row_result[j]
                    
                    except Exception as e:
                        logging.error(f"Error processing batch {batch_idx}: {str(e)}")
                        # Mark all rows in this batch as errors
                        for row_idx in batch_indices:
                            for col in self.output_columns:
                                self.df.at[row_idx, col] = f"Error: {str(e)}"
                    
                    pbar.update(1)
        
        logging.info("Processing complete.")
        return self.df

def read_excel_file(file_path):
    """
    Read an Excel file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the Excel file.
        
    Returns:
        pandas.DataFrame: The dataframe from the Excel file.
    """
    try:
        logging.info(f"Reading Excel file: {file_path}")
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        logging.info(f"Successfully read file with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
        print(f"Error reading file: {str(e)}")
        sys.exit(1)

def validate_columns(df, columns):
    """
    Validate that all columns exist in the dataframe.
    
    Args:
        df (pandas.DataFrame): The dataframe to check.
        columns (list): List of column names to validate.
        
    Returns:
        bool: True if all columns exist, False otherwise.
    """
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing columns: {', '.join(missing_columns)}")
        print(f"Error: The following columns do not exist in the file: {', '.join(missing_columns)}")
        return False
    return True

def get_user_input():
    """
    Get input from the user interactively.
    
    Returns:
        dict: User input parameters.
    """
    print("\n=== Flexible Data Manipulator ===\n")
    
    # Get input file path
    while True:
        input_file = input("Enter path to the input Excel file: ").strip()
        if os.path.isfile(input_file) and input_file.endswith(('.xlsx', '.xls', '.csv')):
            break
        print("Please enter a valid Excel or CSV file path.")
    
    # Read the file
    df = read_excel_file(input_file)
    
    # Show available columns
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    
    # Get input columns
    while True:
        input_cols_str = input("\nEnter the numbers of columns to use as input (comma-separated): ").strip()
        try:
            input_col_indices = [int(idx.strip()) - 1 for idx in input_cols_str.split(',')]
            input_columns = [df.columns[idx] for idx in input_col_indices if 0 <= idx < len(df.columns)]
            
            if not input_columns:
                print("Please select at least one valid column.")
                continue
                
            print(f"Selected input columns: {', '.join(input_columns)}")
            break
        except ValueError:
            print("Please enter valid column numbers separated by commas.")
    
    # Get output columns
    print("\nOutput column options:")
    print("1. Use existing columns")
    print("2. Create new columns")
    
    output_option = input("Select an option (1-2): ").strip()
    
    if output_option == "1":
        # Use existing columns
        while True:
            output_cols_str = input("Enter the numbers of columns to use as output (comma-separated): ").strip()
            try:
                output_col_indices = [int(idx.strip()) - 1 for idx in output_cols_str.split(',')]
                output_columns = [df.columns[idx] for idx in output_col_indices if 0 <= idx < len(df.columns)]
                
                if not output_columns:
                    print("Please select at least one valid column.")
                    continue
                    
                print(f"Selected output columns: {', '.join(output_columns)}")
                break
            except ValueError:
                print("Please enter valid column numbers separated by commas.")
    else:
        # Create new columns
        output_columns = []
        while True:
            col_name = input("Enter a new column name (or press Enter to finish): ").strip()
            if not col_name:
                if not output_columns:
                    print("Please add at least one output column.")
                    continue
                break
            
            if col_name in df.columns:
                overwrite = input(f"Column '{col_name}' already exists. Overwrite? (y/n): ").strip().lower()
                if overwrite != 'y':
                    continue
            
            output_columns.append(col_name)
            print(f"Added output column: {col_name}")
    
    # Get transformation prompt
    print("\nEnter transformation instructions (what to do with the input data):")
    print("Examples:")
    print("1. 'Categorize the product description into one of: Electronics, Clothing, Home Goods, Toys'")
    print("2. 'Summarize the customer feedback in 1-2 sentences'")
    print("3. 'Extract all dates mentioned in the text'")
    print("4. 'Translate the text from English to Spanish'")
    
    transformation_prompt = ""
    print("\nEnter your transformation instructions (type 'END' on a new line when finished):")
    while True:
        line = input()
        if line.strip() == "END":
            break
        transformation_prompt += line + "\n"
    
    if not transformation_prompt.strip():
        print("Please provide transformation instructions.")
        sys.exit(1)
    
    # Get output file path
    output_file = input("\nEnter path for the output Excel file (leave blank for default): ").strip()
    if not output_file:
        # Default to Output files directory with a timestamp in the filename
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUTS_DIR, f"{base_name}_transformed_{timestamp}.xlsx")
    elif not os.path.isabs(output_file):
        # If a relative path is provided, make it relative to the Output files directory
        if not output_file.endswith(('.xlsx')):
            output_file += ".xlsx"
        output_file = os.path.join(OUTPUTS_DIR, output_file)
    elif not output_file.endswith(('.xlsx')):
        output_file += ".xlsx"
    
    # Get model
    models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    print("\nAvailable models:")
    for i, model_name in enumerate(models, 1):
        print(f"{i}. {model_name}")
    
    model = "gpt-3.5-turbo"  # Default
    model_choice = input(f"Select a model (1-{len(models)}, default is 3): ").strip()
    if model_choice:
        try:
            model_idx = int(model_choice) - 1
            if 0 <= model_idx < len(models):
                model = models[model_idx]
        except ValueError:
            pass
    
    # Get temperature
    temperature = 0.3  # Default
    temp_input = input("\nEnter temperature value (0.0-1.0, default is 0.3): ").strip()
    if temp_input:
        try:
            temp_value = float(temp_input)
            if 0.0 <= temp_value <= 1.0:
                temperature = temp_value
        except ValueError:
            pass
    
    # Get batch size
    batch_size = 20  # Default
    batch_input = input("\nEnter batch size (1-50, default is 20): ").strip()
    if batch_input:
        try:
            batch_value = int(batch_input)
            if 1 <= batch_value <= 50:
                batch_size = batch_value
        except ValueError:
            pass
    
    # Check if user wants to limit the number of rows to process (for testing)
    limit_rows = input("\nLimit the number of rows to process? (y/n, default is n): ").strip().lower()
    row_limit = None
    if limit_rows == 'y':
        try:
            row_limit = int(input("Enter the maximum number of rows to process: ").strip())
        except ValueError:
            print("Invalid input. Will process all rows.")
            row_limit = None
    
    return {
        "input_file": input_file,
        "df": df.head(row_limit) if row_limit else df,
        "input_columns": input_columns,
        "output_columns": output_columns,
        "transformation_prompt": transformation_prompt,
        "output_file": output_file,
        "model": model,
        "temperature": temperature,
        "batch_size": batch_size
    }

def main():
    """Main function to run the script."""
    import concurrent.futures
    
    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Manipulate data using OpenAI API.')
        parser.add_argument('--input', required=True, help='Input Excel or CSV file path')
        parser.add_argument('--input-columns', required=True, help='Comma-separated list of input column names')
        parser.add_argument('--output-columns', required=True, help='Comma-separated list of output column names')
        parser.add_argument('--prompt', required=True, help='Transformation prompt (instructions)')
        parser.add_argument('--output', help='Output Excel file path')
        parser.add_argument('--model', default='gpt-3.5-turbo', help='OpenAI model to use')
        parser.add_argument('--temperature', type=float, default=0.3, help='Temperature for generation (0.0-1.0)')
        parser.add_argument('--batch-size', type=int, default=20, help='Batch size for processing')
        parser.add_argument('--limit-rows', type=int, help='Limit the number of rows to process (for testing)')
        
        args = parser.parse_args()
        
        # Read the file
        df = read_excel_file(args.input)
        
        if args.limit_rows:
            df = df.head(args.limit_rows)
        
        # Parse column names
        input_columns = [col.strip() for col in args.input_columns.split(',')]
        output_columns = [col.strip() for col in args.output_columns.split(',')]
        
        # Validate input columns
        if not validate_columns(df, input_columns):
            sys.exit(1)
        
        # Handle output file path
        if args.output:
            if os.path.isabs(args.output):
                output_file = args.output
            else:
                output_file = os.path.join(OUTPUTS_DIR, args.output)
        else:
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(OUTPUTS_DIR, f"{base_name}_transformed_{timestamp}.xlsx")
            
        if not output_file.endswith('.xlsx'):
            output_file += '.xlsx'
        
        # Set up the manipulator
        manipulator = DataManipulator(
            df=df,
            input_columns=input_columns,
            output_columns=output_columns,
            transformation_prompt=args.prompt,
            model=args.model,
            temperature=args.temperature,
            batch_size=args.batch_size
        )
        
    else:
        # Get input interactively
        user_input = get_user_input()
        
        # Set up the manipulator
        manipulator = DataManipulator(
            df=user_input["df"],
            input_columns=user_input["input_columns"],
            output_columns=user_input["output_columns"],
            transformation_prompt=user_input["transformation_prompt"],
            model=user_input["model"],
            temperature=user_input["temperature"],
            batch_size=user_input["batch_size"]
        )
        
        output_file = user_input["output_file"]
    
    # Process the data
    start_time = time.time()
    result_df = manipulator.process_data()
    end_time = time.time()
    
    # Save the result
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save to Excel
        result_df.to_excel(output_file, index=False)
        
        print(f"\nProcessing complete! {len(result_df)} rows processed in {end_time - start_time:.2f} seconds.")
        print(f"Results saved to {output_file}")
        
        # Save a copy of the log with the same base name
        log_output = os.path.splitext(output_file)[0] + "_log.txt"
        try:
            with open(log_file, 'r') as src, open(log_output, 'w') as dst:
                dst.write(src.read())
            print(f"Log saved to {log_output}")
        except Exception as e:
            logging.error(f"Error saving log file: {str(e)}")
            
    except Exception as e:
        logging.error(f"Error saving output file: {str(e)}")
        print(f"Error saving output file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
