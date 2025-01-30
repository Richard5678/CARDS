import json

def remove_prompts_from_json(input_file, output_file):
    try:
        # Read the JSON file
        if input_file.endswith('.json'):    
            with open(input_file, 'r') as f:
                data = json.load(f)
        elif input_file.endswith('.jsonl'):
            with open(input_file, 'r') as f:
                data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported file type: {input_file}")
        
        # If data is a list, process each item
        for item in data:
            # Clean up result by removing prompt prefix if it exists
            item['result'] = item['result'].replace(item['prompt'], '', 1).lstrip()
        
        # Write the modified data back to a new JSON file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Successfully removed prompts and saved to {output_file}")
            
    except FileNotFoundError:
        print(f"Error: Could not find the input file {input_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = '/scratch/ssd004/scratch/rofan'
    input_file = f"{root}/generated_responses_v2_TLDR_2.08_20250129_203526_1.json"  # Replace with your input JSON file name
    output_file = f"{root}/generated_responses_v2_TLDR_2.08_20250129_203526_1_cleaned.json"  # Replace with your desired output file name
    remove_prompts_from_json(input_file, output_file)