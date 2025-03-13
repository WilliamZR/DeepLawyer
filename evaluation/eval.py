from vllm import LLM, SamplingParams
import json
import argparse
import os
import multiprocessing
import multiprocessing.spawn
from utils import BenchmarkProcessor



def run_inference_one_model(gpu_id, data, model_name, sampling_params, tensor_parallel_size):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(','.join(gpu_id))
    llm = LLM(model_name,
              trust_remote_code=True,
              tensor_parallel_size=tensor_parallel_size) 
    
    input_prompts = [d['prompt'] for d in data]
    outputs = llm.generate(input_prompts, sampling_params)

    result = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        result.append({ 'id': data[i]['id'],
                        'source': data[i]['source'],
                        'prompt': prompt, 
                        'generated_text': generated_text,
                        'ground_truth': data[i]['ground_truth']})
                        
    return result
split_list = lambda l, n: [l[i * len(l) // n: (i + 1) * len(l) // n] for i in range(n)]

def run_inference_multi_gpu(model_name, data, sampling_params, tensor_parallel_size=1):
    gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    num_gpus = len(gpu_ids)
    os.environ['WORLD_SIZE'] = str(num_gpus)
    print(f'Using {num_gpus} GPUs')
    
    assert num_gpus % tensor_parallel_size == 0, 'num_gpus should be divisible by tensor_parallel_size to split the data'
    num_processes = num_gpus // tensor_parallel_size
    split_prompts = split_list(data, num_processes)
    split_gpu_ids = split_list(gpu_ids, num_processes)
    data_splits = [(split_gpu_ids[i], p, model_name, sampling_params, tensor_parallel_size) for i, p in enumerate(split_prompts)]
    multiprocessing.set_start_method('spawn', force=True)
    ctx = multiprocessing.get_context('spawn')
    ctx.Process.daemon = False
    with ctx.Pool(processes=num_processes) as pool:
        results = pool.starmap(run_inference_one_model, data_splits)

    outputs = []
    for result in results:
        outputs.extend(result)

    return outputs
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmarks', type=str, default='all')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--force_generate', action = 'store_true')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--system_message', type=str, default='r1-lawyer')
    args = parser.parse_args()


    
    data_processor = BenchmarkProcessor(args.output_dir, args.model_path, args.benchmarks, args.system_message)
    data = data_processor.prepare_all_benchmarks()
    print('**** Prompt Example ****')
    print(data[0]['prompt'])
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    
    outputs = run_inference_multi_gpu(args.model_path, data, sampling_params, tensor_parallel_size=args.tensor_parallel_size)

    
    data_processor.save(outputs, benchmark = '_temp_all_results')

    print('****Example****')
    for key, value in outputs[0].items():
        print(f'{key}: {value}')
    print('=======')
    print('****Evaluation****')
    data_processor.evaluate_all_benchmarks(outputs, save = True)

if __name__ == '__main__':
    main()