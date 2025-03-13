import os
import json
from transformers import AutoTokenizer
import multiprocessing
import re
from rouge import Rouge
import jieba
from bert_score import score as bertscore
import copy
from tqdm import tqdm
from collections import defaultdict

lexeval_subtasks = [
    '1_1', '1_2', '1_3',
    '2_1', '2_2', '2_3', '2_4', '2_5',
    '3_1', '3_2', '3_3', '3_4', '3_5', '3_6',
    '4_1', '4_2',
    '5_1', '5_2', '5_3', '5_4', 
    '6_1', '6_2', '6_3' 
]
benchmark_mapping = {
    #'jec-qa-1-choice_verification': 'data/lawyer/jecqa/JEC_1_choice_verification.json',
    'jec-qa-1-multi-choice': 'data/lawyer/jecqa/JEC_1_multi_choice_test.json',
}
ground_truth_mapping = {
    'jec-qa-1-choice_verification': 'ground_truth',
    'jec-qa-1-multi-choice': 'answer',
}
for subtask in lexeval_subtasks:
    benchmark_mapping[f'lexeval_{subtask}'] = f'data/lawyer/LexEval/processed/{subtask}.jsonl'
    ground_truth_mapping[f'lexeval_{subtask}'] = 'answer'


system_prompt_mapping = {
    'general': 'You are a helpful assistant.',
    'lawyer': '你是一位公正且经验丰富的法律专家，你应该逐步解释法律问题，提供详细的法律建议。',
    'r1-lawyer': (
    "用户和助手之间的对话。用户提出一个问题，由助手来回答。助手首先在脑海中逐步思考推理过程，然后向用户提供答案。推理过程和答案分别用<思考> </思考>和<回答> </回答>标签括起来，即，"
    "<思考> 推理过程 </思考><回答> 答案 </回答>"),
    'r1-distill': (
    "用户和助手之间的对话。用户提出一个问题，由助手来回答。助手首先在脑海中逐步思考推理过程，然后向用户提供答案。推理过程和答案分别用<think> </think>和<answer> </answer>标签括起来，即，"
    "<think> 推理过程 </think><answer> 答案 </answer>"
)
}

class BenchmarkProcessor():
    def __init__(self, output_dir, model_path, benchmarks, system_prompt='general'):
        #self.model_name = model_path.split('/')[-2] if model_path.endswith('/') else model_path.split('/')[-1]
        ## /newdisk/wuzr/models/Qwen2.5-7B-Instruct --> Qwen2.5-7B-Instruct
        ## /newdisk/wuzr/lawyer-o1/checkpoints/qwen2.5-7b-grpo-hard-multiple-choice/global_step_100/actor/huggingface 
        ## --> qwen2.5-7b-grpo-hard-multiple-choice_global_step_100_actor_huggingface
        self.model_name = self.get_model_name(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.benchmarks = self.init_benchmarks(benchmarks) 
        self.system_prompt = system_prompt_mapping[system_prompt]
        self.result_dir = os.path.join(output_dir, self.model_name)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            
    def init_benchmarks(self, benchmarks):
        if benchmarks == 'all':
            return benchmark_mapping.keys()
        else:
            return benchmarks.split(',')
        
    def prepare_all_benchmarks(self):
        all_data = []
        for benchmark in self.benchmarks:
            all_data.extend(self.load(benchmark))
        print('====All data loaded====')
        print(f'Number of data entries: {len(all_data)}')
        return all_data
    
    def evaluate_all_benchmarks(self, output_data, save = True):
        ## split dataset by benchmark_name and evaluate accordingly
        result_dict = {}
        benchmark_data_dict = {}
        benchmark_data_dict = defaultdict(list)
        for entry in output_data:
            benchmark_data_dict[entry['source']].append(entry)
        
        for benchmark, benchmark_data in benchmark_data_dict.items():
            result = self.evaluate(benchmark_data, benchmark, save)
            result_dict[benchmark] = result
        ## summarize the results in a txt file
        with open(os.path.join(self.result_dir, 'result.txt'), 'w', encoding='utf-8') as f:
            for benchmark, result in result_dict.items():
                f.write(f'{benchmark}: {result}\n')
        print('====All benchmarks evaluated====')
        print(f'Results saved in {self.result_dir}')
        for benchmark, result in result_dict.items():
            print(f'{benchmark}: {result}') 
    def get_model_name(self, model_path):            
        if 'checkpoints' in model_path:
            return model_path.split('checkpoints')[-1].replace('/', '_')
        else:
            return model_path.split('/')[-2] if model_path.endswith('/') else model_path.split('/')[-1]
        
    def save(self, data, benchmark):
        with open(os.path.join(self.result_dir, f'{benchmark}.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load(self, benchmark):
        try:    
            input_path = benchmark_mapping[benchmark]
        except KeyError:
            raise ValueError(f'benchmark {benchmark} is not supported')
        
        with open(input_path, 'r', encoding='utf-8') as f:
            if input_path.endswith('.json'):
                data = json.load(f)
            else:
                data = [json.loads(line) for line in f]
        data = self.prepare_prompt_and_answer(data, benchmark_name=benchmark)
        return data

    def prepare_prompt_and_answer(self, data, benchmark_name, num_processors = 16):
        try:
            prompt_func = prompt_func_mapping[benchmark_name]
        except KeyError:
            raise ValueError(f'prompt for benchmark {benchmark_name} is not supported')
        ## add prompt to each data entry as 'prompt' with multiple processing
        
        with multiprocessing.Pool(processes=num_processors) as pool:
            results = pool.starmap(prompt_func, [(entry, self.system_prompt) for entry in data])

        for i, entry in enumerate(data):
            entry['prompt'] = self.tokenizer.apply_chat_template(results[i],tokenize=False, add_generation_prompt=True)
            entry['ground_truth'] = entry[ground_truth_mapping[benchmark_name]]
            entry['source'] = benchmark_name
            if ground_truth_mapping[benchmark_name] != 'ground_truth':
                del entry[ground_truth_mapping[benchmark_name]]
        return data
    
    def evaluate(self, data, benchmark, save = False): 
        try:
            eval_func = eval_func_mapping[benchmark]
        except:
            raise ValueError(f'evaluation for benchmark {benchmark} is not supported')
        eval_results = eval_func(data)
        ## add evaluation results to each data entry
        for i, entry in enumerate(data):
            entry['eval_result'] = eval_results[i]
        if save:
            self.save(data, benchmark)
        ## return average score for score in list
        ## [{accuracy: 0.8}, {rouge-l: 0.6}]
        return {k: sum([entry['eval_result'][k] for entry in data])/len(data) for k in eval_results[0].keys()}
        



def jec_choice_verification_prompt_template(entry, system_prompt,):
    prompt =  JEC_choice_verification_prompt.format(question=entry['statement'], option=entry['option'])
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]
    
def jec_multi_choice_prompt_template(entry, system_prompt):
    prompt = JEC_multi_choice_prompt.format(question=entry['statement'], option_a=entry['option_list']["A"], option_b=entry['option_list']["B"], option_c=entry['option_list']["C"], option_d=entry['option_list']["D"])

    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]

def lexeval_multi_choice_prompt_template(entry, system_prompt):
    prompt = lexeval_prompt_multi_choice.format(question=entry['input'])
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]
    
def lexeval_generation_prompt_template(entry, system_prompt):
    prompt = lexeval_prompt_generation.format(question=entry['input'], instruction=entry['instruction'])
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]

lexeval_prompt_multi_choice = """你是一名法学专家。现在请你解答司法考试中的一道选择题，请你找出所有正确的选项。每道题可能有一个或者多个正确答案。在解答之前，你需要先针对每个提供的选项给出详细的解释。你需要在回答的最后用大括号圈出给出的答案，例如"{{B}}"或者"{{ABD}}"。

选择题：
{question}
"""

lexeval_prompt_generation = """你是一名法学专家。现在请你解答司法考试中的一道问答题。

问题：{instruction}
{question}
"""


JEC_choice_verification_prompt = '''你是一名法学专家。现在请你解答司法考试中的一道题，判断提供的选项是否应该被选择。在解答之前，你需要先针对提供的选项给出详细的解释。你需要在回答的最后直接给出“结论：是”或者“结论：否”。

问题：{question}

选项：{option}'''

JEC_multi_choice_prompt = '''你是一名法学专家。现在请你解答司法考试中的一道选择题，请你找出所有正确的选项。每道题可能有一个或者多个正确答案。在解答之前，你需要先针对每个提供的选项给出详细的解释。你需要在回答的最后用大括号圈出给出的答案，例如"{{B}}"或者"{{ABD}}"。

问题：{question}

选项：
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}'''

prompt_func_mapping = {
    'jec-qa-1-choice_verification': jec_choice_verification_prompt_template,
    'jec-qa-1-multi-choice': jec_multi_choice_prompt_template, 
}

for subtask in lexeval_subtasks:
    if subtask.startswith('5'):
        prompt_func_mapping[f'lexeval_{subtask}'] = lexeval_generation_prompt_template
    else:
        prompt_func_mapping[f'lexeval_{subtask}'] = lexeval_multi_choice_prompt_template

def eval_jec_choice_verification(entry):
    predict = entry['generated_text']
    ground_truth = entry['ground_truth']
    predict = predict.strip()
    
    ### 匹配结论： 是，结果：否，取最后一个，如果没有则返回-1
    result = re.findall(r'结论：(是|否)', predict)
    if result:
        if result[-1] == '是':
            return 1 if ground_truth == 1 else 0
        else:
            return 1 if ground_truth == 0 else 0
    else:
        return 0

def eval_jec_choice_verification_batch(data):
    results = []
    for entry in data:
        results.append(eval_jec_choice_verification(entry))
    return [{'Accuracy': acc} for acc in results] 
    
def eval_jec_multi_choice(entry):
    ## TODO:we should check other multiple choice benchmarks for reference
    ## search for the last { and }
    ## {blabla} {正确A B CC}blabla ==> {A, B, C}
    match = re.search(r'\{[\u4e00-\u9fffA-Za-z,\s]*\}[^{]*$', entry['generated_text'])
    
    if match is None:
        return 0
    ## search for A B C D
    answer = set(re.findall(r'[A-Z]', match.group()))
    ground_truth = set(entry['ground_truth'])
    #print(answer, ground_truth)
    return 1 if answer == ground_truth else 0

def eval_jec_multi_choice_batch(data):
    results = []
    for entry in data:
        results.append(eval_jec_multi_choice(entry))
    return [{'Accuracy': acc} for acc in results]


def eval_lexeval_generation(entry, metrics=['rouge-l', 'bert-score']):
    ground_truth = entry['ground_truth']
    predict = entry['generated_text']
    if '<回答>' in predict and '</回答>' in predict:
    # 提取<回答> </回答>标签中的文本
        _ = re.search(r'<回答>(.*)</回答>', predict, re.DOTALL)
        predict = _.group(1)
    elif '<answer>' in predict and '</answer>' in predict:
        _ = re.search(r'<answer>(.*)</answer>', predict, re.DOTALL)
        predict = _.group(1)
    
    P, R, bertscore_F1 = bertscore([predict], [ground_truth], lang='zh', verbose=True, device='cuda:0')
    
    ground_truth = ' '.join(jieba.cut(ground_truth))
    predict = ' '.join(jieba.cut(predict))
    rouge = Rouge()
    if predict.isspace() or ground_truth.isspace() or len(predict) == 0 or len(ground_truth) == 0:
        rouge_score = 0
    else:
        scores = rouge.get_scores(predict, ground_truth, avg=True)
        rouge_score = scores['rouge-l']['f']
    return {'rouge-l': rouge_score, 'bert-score': bertscore_F1.item()}

def eval_lexeval_generation_batch(data, metrics=['rouge-l']):
    ground_truths = [entry['ground_truth'] for entry in data]
    predicts = [entry['generated_text'] for entry in data]
    for i in range(len(predicts)):
        if '<回答>' in predicts[i] and '</回答>' in predicts[i]:
            _ = re.search(r'<回答>(.*)</回答>', predicts[i], re.DOTALL)
            predicts[i] = _.group(1)
        elif '<answer>' in predicts[i] and '</answer>' in predicts[i]:
            _ = re.search(r'<answer>(.*)</answer>', predicts[i], re.DOTALL)
            predicts[i] = _.group(1)
    ori_predicts = copy.deepcopy(predicts)
    ori_ground_truths = copy.deepcopy(ground_truths)
    results = {}
    for metric in metrics:
        assert metric in ['rouge-l', 'bert-score']
        scores = []
        if metric == 'bert-score':
            P, R, F1 = bertscore(predicts, ground_truths, lang='zh', verbose=True, device='cuda:0', batch_size=64)
            scores = list(F1.cpu().numpy())
            scores = [float(score) for score in scores]
        elif metric == 'rouge-l':
            for i in tqdm(range(len(predicts)), desc='jieba cutting'):
                predicts[i] = ' '.join(jieba.cut(ori_predicts[i]))
                ground_truths[i] = ' '.join(jieba.cut(ori_ground_truths[i]))
            rouge = Rouge()
            for i in tqdm(range(len(predicts)), desc='computing rouge'):
                predict = predicts[i]
                ground_truth = ground_truths[i]
                if predict.isspace() or ground_truth.isspace() or len(predict) == 0 or len(ground_truth) == 0:
                    rouge_score = 0
                else:
                    try:
                        rouge_score = rouge.get_scores(predict, ground_truth, avg=True)["rouge-l"]["f"]
                    except RecursionError:
                        rouge_score = 0
                scores.append(rouge_score)
        
        else:
            raise ValueError(f'Unknown metric: {metric}')
        results[metric] = scores
    results_list = []
    for i in range(len(predicts)):
        results_list.append({metric: results[metric][i] for metric in metrics})
    return results_list


eval_func_mapping = {
    'jec-qa-1-choice_verification': eval_jec_choice_verification_batch,
    'jec-qa-1-multi-choice': eval_jec_multi_choice_batch,
}
## add lexeval subtasks
for subtask in lexeval_subtasks:
    if subtask.startswith('5'):
        eval_func_mapping[f'lexeval_{subtask}'] = eval_lexeval_generation_batch
    else:
        eval_func_mapping[f'lexeval_{subtask}'] = eval_jec_multi_choice_batch