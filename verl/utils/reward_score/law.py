import re
#import jieba
import math
def format_reward(predict):
    """Reward function that checks if the completion has a specific format."""
    pattern = r".*?assistant.*?<思考>.*?</思考>.*?<回答>.*?</回答>"
    match = re.match(pattern, predict, re.DOTALL | re.MULTILINE) 
    return 1.0 if match else 0.0


def verify_multiple_choice(predict, ground_truth):
    ## Hard Reward
    match = re.search(r'\{[\u4e00-\u9fffA-Za-z,\s]*\}[^{]*$', predict)
    
    if match is None:
        return 0
    ## search for A B C D
    answer = set(re.findall(r'[A-Z]', match.group()))
    ground_truth = set(ground_truth)
    return 1 if answer == ground_truth else 0


def compute_score(predict, ground_truth, **kwargs) -> dict:
    """Compute the score of the completion."""
    format_score = format_reward(predict)
    accuracy_score = verify_multiple_choice(predict, ground_truth)
    return {
        "score": format_score + accuracy_score,
        "extra_info": {
            "format_reward": format_score,
            "answer_reward": accuracy_score,
        }
    }
