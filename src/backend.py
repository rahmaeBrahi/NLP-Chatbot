from flask import Flask, render_template, request
from src.main import QAModel
import logging
from pathlib import Path
from datasets import load_dataset

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

qa_model = QAModel()

CONTEXT_FILE = "context.txt"

def load_context_data():
   
    try:
        # Check if context file exists
        if Path(CONTEXT_FILE).exists():
            with open(CONTEXT_FILE, "r", encoding='utf-8') as f:
                return f.read()
        
        logger.info("Loading default context from dataset")
        dataset = load_dataset("cjlovering/natural-questions-short", split="train[:100]")
        context = "\n".join([f"Question: {item['question']}\nAnswer: {item['short_answer']}" 
                           for item in dataset])
        
      
        with open(CONTEXT_FILE, "w", encoding='utf-8') as f:
            f.write(context)
            
        return context
        
    except Exception as e:
        logger.error(f"Error loading context: {e}")
        return """
        This is the default context. Example questions you can ask:
        - What is the capital of France?
        
        """

def chatbot_response(usertext):
    
    try:
        context = load_context_data()
        answer, metadata = qa_model.predict_answer(usertext, context)
        
        if not answer:
            return "I couldn't find an answer to that question in the provided context."
            
        #  confidence 
        if metadata and 'confidence' in metadata:
            return f"{answer} (Confidence: {metadata['confidence']:.0%})"
        return answer
        
    except Exception as e:
        logger.error(f"Error generating chatbot response: {e}")
        return "I'm having trouble processing your request. Please try again later."

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/textchat")
def textchat():
    return render_template('textchat.html')

@app.route("/textchat/get")
def get_bot_response():
    userText = request.args.get('msg')
    if not userText or not userText.strip():
        return "Please enter a valid question."
    return chatbot_response(userText.strip())

@app.route("/about")  
def about():
    return render_template('about.html')  

if __name__ == "__main__":
    
    if not Path(CONTEXT_FILE).exists():
        load_context_data()
    
    app.run(debug=True, host='0.0.0.0', port=5000)