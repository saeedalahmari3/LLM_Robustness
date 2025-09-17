import json
import ollama
import csv
from get_cosine_similarity import *
import os
from tqdm import tqdm 
import torch
from utils import *
import pandas as pd 
from eval_metrics import compute_bleu, compute_rouge, compute_jaccard_index


print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

os.environ["OLLAMA_USE_METAL"] = "1"
# Model name (make sure it's pulled via `ollama pull llama3`)
MODEL_NAME = 'llama3:latest'

print('Listed models...')
print(ollama.list())

# Paths to your JSON files
ORI_JSON_PATH = './data/Amazon_reviews/ori_Keyboard_2999.json'
TRANS_JSON_PATH = './data/Amazon_reviews/trans_Keyboard_2999.json'
if not os.path.exists('ollama'):
    os.makedirs('ollama')

OUTPUT_CSV_PATH = './ollama/'+MODEL_NAME+'_output_results_trans_'+ORI_JSON_PATH.rsplit('/',1)[1].split('.json')[0]+'.csv'

def read_prompts_from_json(json_path):
    prompts = []
    with open(json_path, mode='r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            prompts.append(data['x'])
    return prompts

def generate_responses(prompts):
    responses = []
    for i, text in tqdm(enumerate(prompts, start=1)):
        prompt_template = """Write two sentences at maximum describing the sentiment in the following text:
        
        [DOCUMENT]
        """
        prompt_template = prompt_template.replace("[DOCUMENT]", text)

        if MODEL_NAME == 'llama70b':
            response = llama(prompt_template)
            responses.append(response)
            #print(response)
        elif MODEL_NAME == 'openai':
            response = openai(prompt_template)
            response = response[0]
            responses.append(response)
        else:
            response = ollama.generate(
                model=MODEL_NAME,
                prompt=prompt_template,
                options={'temperature': 0, 'device': 'mps' if torch.backends.mps.is_available() else 'cpu'}
            )
            responses.append(response['response'])

    return responses

def compute_and_save_results(ori_prompts, ori_responses, trans_prompts, trans_responses,OUTPUT_CSV_PATH):
    if not os.path.exists(os.path.basename(OUTPUT_CSV_PATH)):
        print('writing to downloads directory ...')
        OUTPUT_CSV_PATH = "./results/" + MODEL_NAME+'_output_results_trans_'+ORI_JSON_PATH.rsplit('/',1)[1].split('.json')[0]+'.csv'
    with open(OUTPUT_CSV_PATH, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sample ID', 'Original Prompt', 'Original Response', 'Transformed Prompt', 'Transformed Response', 'Cosine Similarity'])

        for i, (ori_prompt, ori_response, trans_prompt, trans_response) in enumerate(zip(ori_prompts, ori_responses, trans_prompts, trans_responses), start=1):
            similarity = get_cosine_similarity(ori_response, trans_response)
            writer.writerow([i, ori_prompt, ori_response, trans_prompt, trans_response, similarity])
            #print(f"Sample {i}: Cosine Similarity = {similarity}")

def evaluate(path2CSV):
    df = pd.read_csv(path2CSV)
    original = df['Original Response'].tolist()
    perturbed = df['Transformed Response'].tolist()

    belu_score = []

    rouge1_precision = []
    rouge1_recall = []
    rouge1_f1 = []

    rouge2_precision = []
    rouge2_recall = []
    rouge2_f1 = []

    rougel_precision = []
    rougel_recall = []
    rougel_f1 = []

    jaccard_index_list = []


    for ori_response, pert_response in zip(original,perturbed):
        belu = compute_bleu(ori_response,pert_response)
        rouge = compute_rouge(ori_response,pert_response)
        jaccard_index = compute_jaccard_index(ori_response,pert_response)
        belu_score.append(belu)
        rouge1_precision.append(rouge[0]['rouge-1']['p'])
        rouge1_recall.append(rouge[0]['rouge-1']['r'])
        rouge1_f1.append(rouge[0]['rouge-1']['f'])

        rouge2_precision.append(rouge[0]['rouge-2']['p'])
        rouge2_recall.append(rouge[0]['rouge-2']['r'])
        rouge2_f1.append(rouge[0]['rouge-2']['f'])

        rougel_precision.append(rouge[0]['rouge-l']['p'])
        rougel_recall.append(rouge[0]['rouge-l']['r'])
        rougel_f1.append(rouge[0]['rouge-l']['f'])

        jaccard_index_list.append(jaccard_index)

    df['BELU'] = belu_score

    df['rouge1-p'] = rouge1_precision
    df['rouge1-r'] = rouge1_recall
    df['rouge1-f1'] = rouge1_f1

    df['rouge2-p'] = rouge2_precision
    df['rouge2-r'] = rouge2_recall
    df['rouge2-f1'] = rouge2_f1

    df['rougel-p'] = rougel_precision
    df['rougel-r'] = rougel_recall
    df['rougel-f1'] = rougel_f1

    df['jaccard_index'] = jaccard_index_list
    df.to_csv(path2CSV.split('.csv')[0]+'_measured.csv',index=False)

if __name__ == '__main__':
    # Read prompts from both original and transformed JSON files
    task = ['eval']
    if 'infer' in task:
        ori_prompts = read_prompts_from_json(ORI_JSON_PATH)
        trans_prompts = read_prompts_from_json(TRANS_JSON_PATH)

        # Generate responses for both original and transformed prompts
        ori_responses = generate_responses(ori_prompts)
        trans_responses = generate_responses(trans_prompts)

        # Compute cosine similarity and save results to a CSV file
        compute_and_save_results(ori_prompts, ori_responses, trans_prompts, trans_responses,OUTPUT_CSV_PATH)
    elif 'eval' in task:
        path2CSV = '/Volumes/Expansion/Collaboration/USF/USF_team_LLM_response_robustness/Final_results_movie0.1'
        for item in os.listdir(path2CSV):
            if item.startswith('._'):
                continue
            elif item.endswith('_measured.csv'):
                continue
            elif item.endswith('.csv'):
                print(item)
                evaluate(os.path.join(path2CSV,item))
            elif os.path.isdir(os.path.join(path2CSV,item)):
                for sub_item in os.listdir(os.path.join(path2CSV,item)):
                    if sub_item.startswith('.'):
                        continue
                    elif sub_item.endswith('_measured.csv'):
                        continue
                    elif sub_item.endswith('.csv'):
                        print(sub_item)
                        evaluate(os.path.join(path2CSV,item,sub_item))

