import dspy
import numpy as np


class Assess(dspy.Signature):
    """Rate the assessed text for compliance with the properties required in the assessment_direction."""
    assessed_text = dspy.InputField()
    assessment_direction = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Scale from 1 to 10")

def extract_score(assessment_answer):
    if len(assessment_answer) <= 2:
        return float(assessment_answer)
    elif 'Assessment Answer:' in assessment_answer:
        return float(assessment_answer.split(':')[-1])
    else:
        print(f"Failure to comply with the required response format: {assessment_answer}")
        return 0


def get_seed_dataset(subset=True):
    with open('data/seed_dataset.json', 'r') as f:
        seed_dataset = json.load(f)

    n = len(seed_dataset['gpt4_bs']["questions"]) if subset else len(seed_dataset['truthful_qa']["questions"])
    seed_dataset = {"questions" : seed_dataset['gpt4_bs']["questions"] + seed_dataset['truthful_qa']["questions"][:n], 
                      "correct_answers" : seed_dataset['gpt4_bs']["correct_answers"] + seed_dataset['truthful_qa']["correct_answers"][:n]
                   }
    
    dataset_dspy = [
        dspy.Example(question=seed_dataset['questions'][i], answer=seed_dataset['correct_answers'][i]).with_inputs("question")
        for i in range(len(seed_dataset['questions']))
        ]
    return dataset_dspy

def get_train_test_set(data, percent=0.3): 
    n = len(data)
    test_size = int(n * percent)
    test_indices = np.random.choice(n, size=test_size, replace=False)
    train_indices = np.setdiff1d(np.arange(n), test_indices)

    cur_trainset = [data[i] for i in train_indices]
    cur_testset = [data[i] for i in test_indices]
    return cur_trainset, cur_testset