'''
Pipeline to run the hard arena.
'''
import os
import yaml
import argparse
import time
from azureml.core import Run

def get_endpoints_key_map(endpoints, is_aml_run):
    endpoint_key_map = {}

    if is_aml_run == "True":
        # for aml run, we get AOAI key from keyvault of the AMl workspace.
        run = Run.get_context()
        ws = run.experiment.workspace
        keyvault = ws.get_default_keyvault()

        for endpoint in endpoints:
            endpt_name = endpoint["name"]
            endpt_key_name = endpoint["name"] + "-aoai-key"
            endpt_key = keyvault.get_secret(endpt_key_name)

            endpoint_key_map[endpt_name] = endpt_key
    else:    
        # for debug local run, add mapping here
        pass

    return endpoint_key_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="Phi-3-mini-4k-instruct"
    )
    parser.add_argument(
        "--model_name", type=str, help="model name corresponds to vllm server"
    )
    parser.add_argument(
        "--judge_model_name", type=str, help="name of the judge model",
        default="tscience-uks-gpt-4o", choices=["tscience-uks-gpt4-1106", "tscience-uks-gpt-4o"]
    )
    parser.add_argument(
        "--baseline_model_name", type=str, help="name of the baseline model",
        default="tscience-uks-gpt-35-turbo-1106", choices=["tscience-uks-gpt-35-turbo-1106", "tscience-uks-gpt-4o"]
    )
    parser.add_argument(
        "--is_aml_run", type=str, default="True", help="if it is an AML run"
    )
    parser.add_argument(
        "--input_dir", type=str, default="None", help="input dir for AML run"
    )
    parser.add_argument(
        "--output_dir", type=str, default="None", help="output dir for AML run"
    )
    parser.add_argument(
        "--port", type=str, default="8008", help="port for hosting vllm"
    )
    parser.add_argument(
        "--max_answer_tokens", type=int, default=2048, help="max tokens for generating answer"
    )
    parser.add_argument(
        "--categories", type=str, default="all", help="categories for generating answer, split by comma"
    )
    args = parser.parse_args()

    # read the api_config file
    file_path = os.path.join(os.path.dirname(__file__), 'config/api_config.yaml')
    with open(file_path, 'r') as file:
        api_config = yaml.safe_load(file)
    
    model_name_list = ["tscience-uks-gpt-35-turbo-1106", "tscience-uks-gpt4-1106", "tscience-uks-gpt-4o"]
    for model_name_api in model_name_list:
        if model_name_api == "tscience-uks-gpt4-1106":
            parallel = 8
        else:
            parallel = 16
        add_dict = {model_name_api: 
                        {'model_name': model_name_api, 
                        'endpoints': [{
                                        'api_base': 'https://aims-oai-research-inference-uks.openai.azure.com/', 
                                        'api_version': '2024-02-01'
                                    }],
                        'api_type': 'azure', 
                        'parallel': parallel,
                        }
                    }
        api_config.update(add_dict)

    # add test model api
    model_id = args.model_id
    model_name = args.model_name

    add_dict = {model_id: 
                    {'model_name': model_name, 
                    'endpoints': [{'api_base': f'http://localhost:{args.port}/v1', 'api_key': 'token-abc123'}], 
                    'api_type': 'openai', 
                    'parallel': 8,
                    }
                }

    api_config.update(add_dict)
    print(api_config)
    file_path = os.path.join(os.path.dirname(__file__), 'config/api_config_test.yaml')
    with open(file_path, 'w') as file:
        yaml.safe_dump(api_config, file, default_flow_style=False)


    # read the gen_answer_config file
    file_path = os.path.join(os.path.dirname(__file__), 'config/gen_answer_config.yaml')
    with open(file_path, 'r') as file:
        gen_answer_config = yaml.safe_load(file)

    gen_answer_config['model_list'] = [model_id]
    gen_answer_config['max_tokens'] = args.max_answer_tokens
    gen_answer_config['categories'] = args.categories.split(',')

    print(gen_answer_config)
    file_path = os.path.join(os.path.dirname(__file__), 'config/gen_answer_config_test.yaml')
    with open(file_path, 'w') as file:
        yaml.safe_dump(gen_answer_config, file, default_flow_style=False)


    # read the judge_config file
    file_path = os.path.join(os.path.dirname(__file__), 'config/judge_config.yaml')
    with open(file_path, 'r') as file:
        judge_config = yaml.safe_load(file)

    judge_config['judge_model'] = args.judge_model_name
    # del judge_config['baseline_model']
    judge_config['baseline'] = False
    judge_config['pairwise'] = False
    judge_config['regex_pattern'] = "\[\[(Pass|Fail)\]\]"
    judge_config['model_list'] = [model_id]
    judge_config['system_prompt'] = '''You are given a math problem along with a reference solution. In the output, you must first extract the final answer from the model Response and compare it with the reference answer. Then you must provide one of the following choices as your final verdict with a label:\n\n1. Model answer is the same as the reference answer: [[Pass]]\n2. Model answer is not the same: [[Fail]]\n\nExample: \"The model answer is XXX. The reference answer is XXX. My final verdict is: [[Pass]]\". Do not inject your own understanding to this problem.'''
    judge_config['prompt_template'] = ["<Math Problem>\n{question_1}\n\n<Model Solution>\n{answer_1}\n<Reference Solution>{reference_1}\n"]
    
    print(judge_config)
    file_path = os.path.join(os.path.dirname(__file__), 'config/judge_config_test.yaml')
    with open(file_path, 'w') as file:
        yaml.safe_dump(judge_config, file, default_flow_style=False)