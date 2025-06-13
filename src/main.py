import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast
from typing import Tuple, Optional, Dict
import logging
from pathlib import Path

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QAModel:
    def __init__(self, model_path: str = "trained_model"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load the trained QA model and tokenizer"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model directory not found at {self.model_path}")
            
            logger.info(f"Loading model from {self.model_path}")
            self.model = BertForQuestionAnswering.from_pretrained(self.model_path)
            self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def preprocess_inputs(
        self,
        question: str,
        context: str,
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        
        return self.tokenizer(
            question,
            context,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=False  
        ).to(self.device)

    def predict_answer(
        self,
        question: str,
        context: str,
        max_length: int = 512,
        max_answer_length: int = 200
    ) -> Tuple[Optional[str], Optional[Dict]]:
        
        try:
            # input validation
            question = question.strip()
            context = context.strip()
            if not question or not context:
                raise ValueError("Question and context cannot be empty")
            
            inputs = self.preprocess_inputs(question, context, max_length)
            
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # process scores
            start_scores = torch.nn.functional.softmax(outputs.start_logits, dim=1)
            end_scores = torch.nn.functional.softmax(outputs.end_logits, dim=1)
            
            start_idx = torch.argmax(start_scores).item()
            end_idx = torch.argmax(end_scores).item() + 1  # +1 for inclusive end
            
            # validate positions
            seq_len = inputs["input_ids"].shape[1]
            if start_idx >= seq_len or end_idx > seq_len or start_idx >= end_idx:
                logger.warning("Invalid answer span positions")
                return None, None
            
            # answer
            answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
            answer_text = self.tokenizer.decode(
                answer_tokens,
                skip_special_tokens=True
            ).strip()
            
            # post-processing
            answer_text = self._clean_answer_text(answer_text)
            if not answer_text:
                return None, None
                
            if len(answer_text) > max_answer_length:
                answer_text = answer_text[:max_answer_length] + "..."
            
            # confidence metrics
            confidence = (start_scores[0, start_idx].item() + 
                         end_scores[0, end_idx-1].item()) / 2
            
            metadata = {
                'confidence': round(confidence, 4),
                'start_pos': start_idx,
                'end_pos': end_idx,
                'input_length': seq_len
            }
            
            return answer_text, metadata
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            return None, None

    def _clean_answer_text(self, text: str) -> str:
        """Clean answer text from tokenization """
        text = ' '.join(text.split()) 
        for punct in [',', '.', '?', '!']:
            text = text.replace(f' {punct}', punct)
        return text.capitalize()

def main():
    
    try:
        qa = QAModel()
        
        context = """
        Google Colab is a free cloud service that allows users to run Python code 
        in their browsers. It provides access to GPUs and is widely used for 
        machine learning projects.
        """
        
        questions = [
            "What is Google Colab?",
            "What kind of service is Colab?",
            "Does Colab provide GPU access?"
        ]
        
        for q in questions:
            answer, meta = qa.predict_answer(q, context)
            print(f"\nQ: {q}")
            print(f"A: {answer if answer else 'No answer found'}")
            if meta:
                print(f"Confidence: {meta['confidence']:.1%}")
                
    except Exception as e:
        logger.critical(f"Application error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())