# import os
# import torch
# import time
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import evaluate
# from transformers import BertForQuestionAnswering, BertTokenizerFast
# from torch.optim import AdamW

# # Configuration
# os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# MODEL_NAME = "bert-base-uncased"
# BATCH_SIZE = 8  # Increased from 2
# MAX_LENGTH = 384  # Increased from 256
# NUM_EPOCHS = 5   # Increased from 3
# LEARNING_RATE = 3e-5  # Adjusted from 2e-5

# def prepare_data(examples, tokenizer):
#     # Process questions
#     questions = [q[0]['input_text'] if q and isinstance(q, list) and len(q) > 0 else "" 
#                 for q in examples['questions']]
    
#     # Tokenize inputs
#     inputs = tokenizer(
#         questions,
#         examples['contexts'],
#         truncation=True,
#         max_length=MAX_LENGTH,
#         padding='max_length',
#         return_tensors='pt',
#         return_offsets_mapping=True
#     )
    
#     # Process answer positions
#     start_positions = []
#     end_positions = []
#     offset_mapping = inputs.pop('offset_mapping')
    
#     for i, answer_list in enumerate(examples['answers']):
#         start_pos = end_pos = 0
#         if answer_list and isinstance(answer_list, list) and len(answer_list) > 0:
#             answer = answer_list[0]
#             answer_start = answer.get('span_start', 0)
#             answer_end = answer.get('span_end', 1)
            
#             # Validate answer positions
#             if answer_start >= len(examples['contexts'][i]) or answer_end > len(examples['contexts'][i]):
#                 start_positions.append(0)
#                 end_positions.append(0)
#                 continue
                
#             for token_idx, (start, end) in enumerate(offset_mapping[i]):
#                 if start <= answer_start < end:
#                     start_pos = token_idx
#                 if start < answer_end <= end:
#                     end_pos = token_idx
#                     break
        
#         start_positions.append(start_pos)
#         end_positions.append(end_pos)
    
#     return {
#         'input_ids': inputs['input_ids'],
#         'attention_mask': inputs['attention_mask'],
#         'start_positions': torch.tensor(start_positions),
#         'end_positions': torch.tensor(end_positions)
#     }

# def collate_fn(batch):
#     return {
#         'input_ids': torch.stack([item['input_ids'] for item in batch]),
#         'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
#         'start_positions': torch.stack([item['start_positions'] for item in batch]),
#         'end_positions': torch.stack([item['end_positions'] for item in batch])
#     }

# def main():
#     try:
#         # 1. Load and inspect dataset
#         print("1/5 Loading and inspecting dataset...")
#         start_time = time.time()
#         dataset = load_dataset("cjlovering/natural-questions-short", split="train")
        
#         # Print sample data for inspection
#         print("\nSample data inspection:")
#         print("Question:", dataset['questions'][0][0]['input_text'])
#         print("Answer:", dataset['answers'][0][0])
#         print("Context:", dataset['contexts'][0][:100] + "...")
        
#         dataset = dataset.train_test_split(test_size=0.2)
#         print(f"\nDataset loaded in {time.time()-start_time:.2f} seconds")

#         # 2. Initialize tokenizer
#         print("\n2/5 Loading tokenizer...")
#         tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
#         if tokenizer.pad_token is None:
#             tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#         # 3. Initialize model
#         print("\n3/5 Loading model...")
#         model = BertForQuestionAnswering.from_pretrained(MODEL_NAME)

#         # 4. Prepare data
#         print("\n4/5 Preparing data...")
#         train_dataset = dataset['train'].map(
#             lambda x: prepare_data(x, tokenizer),
#             batched=True,
#             batch_size=100,
#             remove_columns=dataset['train'].column_names
#         )
#         eval_dataset = dataset['test'].map(
#             lambda x: prepare_data(x, tokenizer),
#             batched=True,
#             batch_size=100,
#             remove_columns=dataset['test'].column_names
#         )

#         # Create DataLoaders
#         train_loader = DataLoader(
#             train_dataset.with_format("torch"),
#             batch_size=BATCH_SIZE,
#             shuffle=True,
#             collate_fn=collate_fn
#         )
#         eval_loader = DataLoader(
#             eval_dataset.with_format("torch"),
#             batch_size=BATCH_SIZE,
#             collate_fn=collate_fn
#         )

#         # Training setup
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         model.to(device)
#         optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

#         # 5. Training loop
#         print("\n5/5 Starting training...")
#         def train_model():
#             model.train()
#             for epoch in range(NUM_EPOCHS):
#                 total_loss = 0
#                 progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
                
#                 for batch in progress_bar:
#                     inputs = {
#                         'input_ids': batch['input_ids'].to(device),
#                         'attention_mask': batch['attention_mask'].to(device),
#                         'start_positions': batch['start_positions'].to(device),
#                         'end_positions': batch['end_positions'].to(device)
#                     }
                    
#                     optimizer.zero_grad()
#                     outputs = model(**inputs)
#                     loss = outputs.loss
#                     loss.backward()
#                     optimizer.step()
                    
#                     total_loss += loss.item()
#                     progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
#                 print(f"Epoch {epoch + 1} completed. Average loss: {total_loss/len(train_loader):.4f}")

#         def evaluate_model():
#             model.eval()
#             metric = evaluate.load("squad_v2")  # Changed to squad_v2
            
#             for batch in tqdm(eval_loader, desc="Evaluating"):
#                 inputs = {
#                     'input_ids': batch['input_ids'].to(device),
#                     'attention_mask': batch['attention_mask'].to(device)
#                 }
                
#                 with torch.no_grad():
#                     outputs = model(**inputs)
                
#                 start_pred = outputs.start_logits.argmax(dim=1)
#                 end_pred = outputs.end_logits.argmax(dim=1)
                
#                 predictions = [
#                     {'prediction_text': tokenizer.decode(
#                         batch['input_ids'][i][start_pred[i]:end_pred[i]+1],
#                         skip_special_tokens=True
#                     ), 'id': str(i)}
#                     for i in range(len(batch['input_ids']))
#                 ]
                
#                 references = [
#                     {'answers': {
#                         'text': [tokenizer.decode(
#                             batch['input_ids'][i][batch['start_positions'][i]:batch['end_positions'][i]+1],
#                             skip_special_tokens=True
#                         )],
#                         'answer_start': [0]
#                     }, 'id': str(i)}
#                     for i in range(len(batch['input_ids']))
#                 ]
                
#                 metric.add_batch(predictions=predictions, references=references)
            
#             results = metric.compute()
#             print(f"\nEvaluation Results:\nExact Match: {results['exact_match']:.2f}%\nF1 Score: {results['f1']:.2f}")

#         # Run training and evaluation
#         train_model()
#         evaluate_model()

#         # Save model
#         print("\nSaving model...")
#         model.save_pretrained("trained_model")
#         tokenizer.save_pretrained("trained_model")
#         print("Training completed successfully!")

#     except Exception as e:
#         print(f"\nError occurred: {str(e)}")
#         print("Troubleshooting suggestions:")
#         print("1. Check data format and preprocessing")
#         print("2. Verify answer spans are correct")
#         print("3. Try smaller subset of data first")

# if __name__ == "__main__":
#     if os.name == 'nt':
#         import multiprocessing
#         multiprocessing.freeze_support()
    
#     main()
import os
import torch
import time
import json
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
from transformers import BertForQuestionAnswering, BertTokenizerFast
from torch.optim import AdamW
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_NAME = "deepset/bert-base-cased-squad2"
BATCH_SIZE = 16
MAX_LENGTH = 384
NUM_EPOCHS = 5
LEARNING_RATE = 3e-5
ERROR_SAMPLES_FILE = "error_samples.jsonl"

def validate_answer_span(context: str, answer: dict) -> bool:
    """Validate answer span positions and text matching"""
    start = answer['span_start']
    end = answer['span_end']
    
    if start >= len(context) or end > len(context):
        return False
        
    if context[start:end] != answer['span_text']:
        return False
        
    return True

def log_error_sample(context, question, answer, error_type):
    """log error samples for further analysis"""
    with open(ERROR_SAMPLES_FILE, "a", encoding='utf-8') as f:
        f.write(json.dumps({
            "error_type": error_type,
            "context": context,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }) + "\n")

def prepare_data(examples, tokenizer):
    """enhanced data preparation """
    questions = [q[0]['input_text'] if q and isinstance(q, list) and len(q) > 0 else "" 
                for q in examples['questions']]
    
    inputs = tokenizer(
        questions,
        examples['contexts'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        return_tensors='pt',
        return_offsets_mapping=True
    )
    
    start_positions = []
    end_positions = []
    offset_mapping = inputs.pop('offset_mapping')
    invalid_count = 0
    
    for i, answer_list in enumerate(examples['answers']):
        start_pos = end_pos = 0
        if answer_list and isinstance(answer_list, list) and len(answer_list) > 0:
            answer = answer_list[0]
            
            if not validate_answer_span(examples['contexts'][i], answer):
                log_error_sample(
                    examples['contexts'][i],
                    questions[i],
                    answer,
                    "invalid_span"
                )
                invalid_count += 1
                start_positions.append(0)
                end_positions.append(0)
                continue
                
            answer_start = answer['span_start']
            answer_end = answer['span_end']
            
            for token_idx, (start, end) in enumerate(offset_mapping[i]):
                if start <= answer_start < end:
                    start_pos = token_idx
                if start < answer_end <= end:
                    end_pos = token_idx
                    break
        
        start_positions.append(start_pos)
        end_positions.append(end_pos)
    
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count}/{len(examples['answers'])} invalid answer spans")
    
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'start_positions': torch.tensor(start_positions),
        'end_positions': torch.tensor(end_positions)
    }

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'start_positions': torch.stack([item['start_positions'] for item in batch]),
        'end_positions': torch.stack([item['end_positions'] for item in batch])
    }

def inspect_sample_data(dataset, tokenizer, num_samples=3):
    """ sample data for validation"""
    logger.info("\nData Sample Inspection:")
    for i in range(num_samples):
        sample = dataset[i]
        question = sample['questions'][0]['input_text']
        context = sample['contexts']
        answer = sample['answers'][0]
        
        logger.info(f"\nSample {i + 1}:")
        logger.info(f"Question: {question}")
        logger.info(f"Context: {context[:200]}...")
        logger.info(f"Answer: {answer['span_text']}")
        logger.info(f"Start: {answer['span_start']}, End: {answer['span_end']}")
        
        # answer span
        if not validate_answer_span(context, answer):
            logger.warning("Invalid answer span detected!")

def main():
    try:
       
        if Path(ERROR_SAMPLES_FILE).exists():
            os.remove(ERROR_SAMPLES_FILE)
        
        logger.info("1/5 Loading and inspecting dataset...")
        dataset = load_dataset("cjlovering/natural-questions-short", split="train")
        inspect_sample_data(dataset, None)
        
        dataset = dataset.train_test_split(test_size=0.2)
        
        logger.info("\n2/5 Loading tokenizer...")
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        #  model
        logger.info("\n3/5 Loading model...")
        model = BertForQuestionAnswering.from_pretrained(MODEL_NAME)
        
        logger.info("\n4/5 Preparing data...")
        train_dataset = dataset['train'].map(
            lambda x: prepare_data(x, tokenizer),
            batched=True,
            batch_size=100,
            remove_columns=dataset['train'].column_names
        )
        eval_dataset = dataset['test'].map(
            lambda x: prepare_data(x, tokenizer),
            batched=True,
            batch_size=100,
            remove_columns=dataset['test'].column_names
        )
        
        # create DataLoaders
        train_loader = DataLoader(
            train_dataset.with_format("torch"),
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        eval_loader = DataLoader(
            eval_dataset.with_format("torch"),
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn
        )
        
     
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        
        #  training loop
        logger.info("\n5/5 Starting training...")
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
            
            for batch in progress_bar:
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'start_positions': batch['start_positions'].to(device),
                    'end_positions': batch['end_positions'].to(device)
                }
                
                optimizer.zero_grad()
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {total_loss/len(train_loader):.4f}")
        
        # Save model
        logger.info("\nSaving model...")
        model.save_pretrained("trained_model")
        tokenizer.save_pretrained("trained_model")
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()