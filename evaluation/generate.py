from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
system_prompt =  (
    "用户和助手之间的对话。用户提出一个问题，由助手来回答。助手首先在脑海中逐步思考推理过程，然后向用户提供答案。推理过程和答案分别用<思考> </思考>和<回答> </回答>标签括起来，即，"
    "<思考> 推理过程 </思考><回答> 答案 </回答>")
if __name__ == '__main__':
    model_name = '/home/wuzr/lawyer-o1/checkpoints/qwen2.5-7b-grpo-hard-multiple-choice/global_step_100/actor/huggingface'
    model = LLM(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_text = '''你是一名法学专家。现在请你解答司法考试中的一道选择题，请你找出所有正确的选项。每道题可能有一个或者多个正确答案。在解答之前，你需要先针对每个提供的选项给出详细的解释。你需要在回答的最后用大括号圈出给出的答案，例如\"{{B}}\"或者\"{{ABD}}\"。
    
    问题：甲公司与乙公司就双方签订的加工承揽合同达成仲裁协议，约定一旦合同履行发生纠纷，由当地仲裁委员会仲裁。后合同履行发生争议，甲公司将乙公司告上法庭。对此乙公司没有向受诉法院提出异议。开庭审理中，甲公司举出充分证据，乙公司败诉几成定局，于是乙公司向法院提交了双方达成的仲裁协议。法院审查后认为该仲裁协议无效，此时应如何处理?
    
    选项：
    A: 继续审理
    B: 判决该仲裁协议无效
    C: 如甲公司对仲裁协议效力没有异议，则裁定驳回起诉
    D: 将仲裁协议的效力问题移交有关仲裁委员会审理。
'''
    prompt = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': input_text}
    ]
    input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=2048)
    output = model.generate(input_text, sampling_params)
    print(output[0].outputs[0].text)