import pandas as pd
import numpy as np
import plotly.express as px

import tiktoken
import datetime
import argparse
import os
import math
import re

from glob import glob
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from utils import load_model_answers
import jsonlines
from tabulate import tabulate
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="arena-hard-v0.1")
    parser.add_argument("--judge-name", type=str, default="gpt-4-1106-preview")
    args = parser.parse_args()

    data_dir = os.path.join(f"data/{args.bench_name}/model_judgment", args.judge_name)
    data_files = glob(os.path.join(data_dir, "*.jsonl"))

    language_tier = {"Tier-1": ["Chinese", "Japanese", "French", "Spanish", "German", "Italian", "Portuguese"], "Tier-2": ["Arabic", "Hindi", "Korean", "Czech", "Danish", "Finnish", "Hebrew", "Hungarian", "Dutch", "Norwegian", "Polish", "Russian", "Swedish", "Thai", "Turkish", "Ukrainian"]}

    for data_file in data_files:
        model_name = data_file.split('/')[-1].split('.')[0]
        print(f"================== {model_name} ====================")

        # Parse the JSONL data
        json_data = []
        with jsonlines.open(data_file) as reader:
            for obj in reader:
                json_data.append(obj)
        
        # load model answer
        answer_dir = f"data/{args.bench_name}/model_answer"
        with jsonlines.open(os.path.join(answer_dir, f"{model_name}.jsonl")) as reader:
            model_answers = {entry['question_id']: entry['choices'] for entry in reader}

        # Initialize a dictionary to store pass and fail rates
        results = {}

        # Process each entry in the JSONL data
        for entry in json_data:
            # Extract language and category
            question_id = entry['question_id']
            language = question_id.split('[')[1].split(']')[0]
            category = entry['category']
            
            # Extract the score
            score = entry['games'][0]['score']
            
            # Initialize the language and category in the results dictionary if not already present
            if language not in results:
                results[language] = {}
            if category not in results[language]:
                results[language][category] = {'Pass': 0, 'Fail': 0, 'Missing': 0, "AVG len": []}
            
            # Update the pass or fail count
            if score == 'Pass':
                results[language][category]['Pass'] += 1
            elif score == 'Fail':
                results[language][category]['Fail'] += 1
            elif isinstance(score, float):
                results[language][category]['Pass'] += score
            else:
                results[language][category]['Missing'] += 1
            results[language][category]["AVG len"].append(model_answers[question_id][0]["turns"][0]["token_len"])

        # Print the results beautifully
        table_data = []
        tier_rate = {}
        tier_rate["Tier-1"] = defaultdict(dict)
        tier_rate["Tier-2"] = defaultdict(dict)
        for language, categories in results.items():
            for category, counts in categories.items():
                total = len(counts["AVG len"])
                pass_rate = counts['Pass'] / total * 100 if total > 0 else 0
                fail_rate = counts['Fail'] / total * 100 if total > 0 else 0
                missing_rate = counts['Missing'] / total * 100 if total > 0 else 0
                avg_len = np.mean(counts["AVG len"])
                table_data.append([language, category, f"{pass_rate:.2f}%", f"{fail_rate:.2f}%", f"{missing_rate:.2f}%", f"{avg_len:.0f}"])

                for tier, languages in language_tier.items():
                    if language in languages:
                        if category not in tier_rate[tier]:
                            tier_rate[tier][category] = {'pass_rate': [], 'fail_rate': [], 'missing_rate': [], 'avg_len': []}
                        tier_rate[tier][category]["pass_rate"].append(pass_rate)
                        tier_rate[tier][category]["fail_rate"].append(fail_rate)
                        tier_rate[tier][category]["missing_rate"].append(missing_rate)
                        tier_rate[tier][category]["avg_len"].append(avg_len)
        for tier, categories in tier_rate.items():
            for category, rates in categories.items():
                pass_rate = np.mean(rates["pass_rate"])
                fail_rate = np.mean(rates["fail_rate"])
                missing_rate = np.mean(rates["missing_rate"])
                avg_len = np.mean(rates["avg_len"])
                table_data.append([tier, category, f"{pass_rate:.2f}%", f"{fail_rate:.2f}%", f"{missing_rate:.2f}%", f"{avg_len:.0f}"])
        print(tabulate(table_data, headers=["Language", "Category", "Pass Rate", "Fail Rate", "Missing Rate", "AVG len"]))

        # save examples
        sorted_examples = defaultdict(lambda: defaultdict(list))

        # Process each entry in the JSONL data
        for entry in json_data:
            # Extract language and category
            question_id = entry['question_id']
            language = question_id.split('[')[1].split(']')[0]
            category = entry['category']    
            sorted_examples[language][category].append(entry)

        # Save the failed examples to separate files for each language with pretty-printing
        for language, categories in sorted_examples.items():
            output_file = os.path.join(data_dir, model_name, f"examples_{language}.jsonl")
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            with open(output_file, mode='w') as writer:
                for category, examples in sorted(categories.items()):
                    for example in examples:
                        writer.write(json.dumps(example, indent=2, ensure_ascii=False) + '\n') 