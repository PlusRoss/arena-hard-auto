#!/bin/bash

# Command line arguments with defaults
question_file="${1:-question.jsonl}"
input_dir="${2:-None}"
output_dir="${3:-None}"

# Check if input directory is not 'None' and copy data from it
if [ "$input_dir" != "None" ]; then
    cp -r "$input_dir"/arena-hard-v0.1 data/
fi

# Run judgment generation
python gen_judgment.py --setting-file config/judge_config_test.yaml --endpoint-file config/api_config_test.yaml --question-file $question_file

# load config/judge_config_test.yaml and extract judge_name
judge_name=$(python -c "import yaml; print(yaml.safe_load(open('config/judge_config_test.yaml'))['judge_model'])")

# Show results
python show_result.py --judge-name "$judge_name"

# Copy results to output dir if not 'None'
if [ "$output_dir" != "None" ]; then
    cp -r data/arena-hard-v0.1 "$output_dir"
fi
