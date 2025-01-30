import json

def convert_jsonl_to_json(input_file, output_file, read_every_n_lines=1):
    # List to store all converted entries
    converted_data = []
    
    # Read the JSONL file line by line
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line_idx % read_every_n_lines != 0:
                continue

            entry = json.loads(line.strip())
                
            # Create new dictionary with desired field mapping
            converted_entry = {
                "prompt": entry.get("prompt", ""),
                "result": entry.get("response", "").replace(entry.get("prompt", ""), "", 1).lstrip()
            }
            
            converted_data.append(converted_entry)
    
    # Write the converted data to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    root = "/scratch/ssd004/scratch/rofan"
    # input_file = f"{root}/generated_responses_v2_TLDR_2.08_20250129_203526_1.jsonl"  # Change this to your input file name
    # output_file = f"{root}/generated_responses_v2_TLDR_2.08_20250129_203526_1.json"  # Change this to your desired output file name
    
    input_file = f"{root}/generated_responses_HH_1.73_20250128_235908_10.jsonl"
    output_file = f"{root}/generated_responses_HH_1.73_20250128_235908_10_cleaned.json"
    
    try:
        convert_jsonl_to_json(input_file, output_file, read_every_n_lines=10)
        print(f"Successfully converted {input_file} to {output_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
