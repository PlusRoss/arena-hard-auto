#!/bin/bash

# Pass the model name and port as command line arguments to this script
model_name="$1"
port="${2:-8019}"  # Default port value is 8019
question_file="${3:-question.jsonl}" # Default question_file value is 'question.jsonl'
output_dir="${4:-None}"  # Default output_dir value is 'None'

model_max_length=$(python3 -c "import json; print(json.load(open('$model_name/config.json'))['max_position_embeddings'])")
# vllm max length is 8192 if model max length is greater than 8192 or model_max_length does not exist
vllm_max_length=$((model_max_length > 8192 || model_max_length == null ? 8192 : model_max_length))
echo "Model max length: $model_max_length, VLLM max length: $vllm_max_length"

# Start the model vllm hosting
# nohup python -m vllm.entrypoints.openai.api_server --model "$model_name" --max-model-len $vllm_max_length --dtype auto --api-key token-abc123 --port "$port" --trust-remote-code > data/arena-hard-v0.1/server_output.log 2>&1 &
python -m vllm.entrypoints.openai.api_server --model "$model_name" --max-model-len $vllm_max_length --dtype auto --api-key token-abc123 --port "$port" --trust-remote-code &

# Wait for the server to start
sleep 60

# Run answer generation 
python gen_answer.py --setting-file config/gen_answer_config_test.yaml --endpoint-file config/api_config_test.yaml --question-file "$question_file"

# Stop the model vllm hosting
pkill -f vllm.entrypoints.openai.api_server

# Wait for the server to stop
sleep 10

# Copy results to output dir if not 'None'
if [ "$output_dir" != "None" ]; then
  cp -r data/arena-hard-v0.1 "$output_dir"
fi
