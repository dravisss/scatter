import pandas as pd
import sys

def clean_csv(input_file, output_file):
    # Read the file as text first
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Clean the lines
    cleaned_lines = []
    for line in lines:
        # Remove trailing semicolon and whitespace
        line = line.rstrip(';').rstrip()
        cleaned_lines.append(line)
    
    # Write cleaned file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines))
    
    print(f"Cleaned file saved as: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_csv.py input_file.csv output_file.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_csv(input_file, output_file)
