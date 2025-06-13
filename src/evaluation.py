import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import numpy as np


MODEL_PATH = "trained_model"
MAX_LENGTH = 384
NUM_EXAMPLES = 10  
def evaluate_model():
    
    try:
        
        dataset = load_dataset("cjlovering/natural-questions-short", split="train")
        test_samples = dataset.select(range(NUM_EXAMPLES))
        
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
        model = BertForQuestionAnswering.from_pretrained(MODEL_PATH)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        #  initialize metrics and storae
        metric = evaluate.load("squad")
        exact_matches = []
        f1_scores = []
        precisions = []
        recalls = []

        
        for example in tqdm(test_samples, desc="Evaluating"):
            # tokenize and prepare input
            question = example['questions'][0]['input_text'] if example['questions'] else ""
            inputs = tokenizer(
                question,
                example['contexts'],
                truncation=True,
                max_length=MAX_LENGTH,
                padding='max_length',
                return_tensors='pt',
                return_offsets_mapping=True
            ).to(device)
            
            #  prediction
            with torch.no_grad():
                outputs = model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
            
            start_pred = outputs.start_logits.argmax().item()
            end_pred = outputs.end_logits.argmax().item() + 1
            
            #  answer
            answer_text = ""
            if start_pred < len(inputs['offset_mapping'][0]) and end_pred <= len(inputs['offset_mapping'][0]):
                start_char = inputs['offset_mapping'][0][start_pred][0].item()
                end_char = inputs['offset_mapping'][0][end_pred-1][1].item()
                answer_text = example['contexts'][start_char:end_char]
            
         
            prediction = {'prediction_text': answer_text, 'id': "0"}
            
            answer_list = example['answers']
            if answer_list and isinstance(answer_list, list) and len(answer_list) > 0:
                answer = answer_list[0]
                reference = {
                    'answers': {'text': [answer['span_text']], 'answer_start': [answer['span_start']]},
                    'id': "0"
                }
            else:
                reference = {
                    'answers': {'text': [""], 'answer_start': [0]},
                    'id': "0"
                }
            
            # calculate metrics
            result = metric.compute(predictions=[prediction], references=[reference])
            exact_matches.append(result['exact_match'])
            f1_scores.append(result['f1'])
            
            # calculate precision and recall
            pred_tokens = set(answer_text.lower().split())
            true_tokens = set(reference['answers']['text'][0].lower().split())
            
            if len(pred_tokens) > 0:
                precision = len(pred_tokens & true_tokens) / len(pred_tokens)
            else:
                precision = 0.0
                
            if len(true_tokens) > 0:
                recall = len(pred_tokens & true_tokens) / len(true_tokens)
            else:
                recall = 0.0
                
            precisions.append(precision)
            recalls.append(recall)

        
        avg_exact_match = np.mean(exact_matches)
        avg_f1 = np.mean(f1_scores)
        avg_precision = np.mean(precisions) * 100  
        avg_recall = np.mean(recalls) * 100       
        
        print("\nEvaluation Results:")
        print(f" Exact Match: {avg_exact_match:.2f}%")
        print(f" F1 Score: {avg_f1:.2f}%") 
        print(f" Precision: {avg_precision:.2f}%")
        print(f" Recall: {avg_recall:.2f}%")
        
        return {
            'exact_match': avg_exact_match,
            'f1': avg_f1,
            'precision': avg_precision,
            'recall': avg_recall
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_model()