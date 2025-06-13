# NLP Chatbot: Contextual Question-Answering System

An intelligent chatbot system powered by BERT NLP model that answers questions based on user-provided context with high accuracy and interactive web interface.

##  Overview

This project implements a sophisticated NLP chatbot that can answer questions based on context provided by users. The system utilizes the **deepset/bert-base-cased-squad2** model, a variant of BERT (Bidirectional Encoder Representations from Transformers) that has been fine-tuned for question-answering tasks. The model was further trained on the Natural Questions dataset to enhance its performance on contextual question answering.

##  Key Features

- **BERT-based Q&A**: Utilizes `deepset/bert-base-cased-squad2` model for accurate answer extraction
- **Interface**: Web-based chat interface with responsive design
- **High Accuracy**: Achieved 80.11% F1 score on validation data
- **Easy Integration**: Flask-based backend with Bootstrap frontend
- **Real-time Processing**: Fast tokenization and answer prediction
- **Context-aware**: Extracts precise answers from provided text passages

## Technical Specifications

### Model Performance
| Metric | Score |
|--------|-------|
| F1 Score | 80.11% |
| Precision | 77.50% |
| Recall | 88.57% |

### Technology Stack

**Backend:**
- Transformers (BERT Q&A)
- PyTorch
-  Flask


**Frontend:**
-  Bootstrap 5
-  JavaScript
-  Responsive design

**Model Architecture:**
- **Base Model**: deepset/bert-base-cased-squad2
- **Tokenizer**: BertTokenizerFast
- **Training Dataset**: Natural Questions (short answers)
- **Fine-tuning**: SQuAD dataset optimization

##  Project Structure

```
NLP_CHATBOT/
├── notebooks/
│   └── NLP_Chatbot_training_&evaluation.ipynb
├── src/
│   ├── backend.py
│   ├── evaluation.py
│   ├── main.py
│   └── train_evaluation.py
├── static/
│   ├── assets/
│   ├── css/
│   ├── js/
│   └── uploads/
├── templates/
│   ├── about.html
│   ├── index.html
│   ├── layout.html
│   └── textchat.html
├── trained_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
├── config.json
├── README.md
└── requirements.txt
```

##  Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/rahmaeBrahi/Chatbot-NLP.git
   cd Chatbot-NLP
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the trained model**
   
   The pre-trained model files are not included in the repository due to size constraints. You can download the trained model from the following link:
   
   **📥 [Download Trained Model](https://drive.google.com/drive/folders/166pRH6aaMfyaAy2ph1JUETRTKk4D0fkz?usp=sharing)**
   
   After downloading:
   - Extract the model files to the `trained_model/` directory
   - Ensure all model files (config.json, model.safetensors, tokenizer files, etc.) are placed in the correct location

4. **Initialize the model**
   ```bash
   python main.py --init-model
   ```

5. **Run the backend server**
   ```bash
   python src/backend.py
   ```

6. **Access the application**
   - Open your web browser and navigate to `http://localhost:5000`
   - Use the text chat interface to interact with the chatbot

## 💡 Usage

### Basic Usage
1. **Provide Context**: Enter a paragraph or passage of text that contains the information
2. **Ask Question**: Type your question related to the provided context
3. **Get Answer**: The chatbot will analyze the context and extract the most relevant answer

### Example Interaction
**Context:** 
```
The Eiffel Tower, built by Gustave Eiffel, is a famous landmark in Paris, France. 
It was completed in 1889 and stands 324 meters tall.
```

**Question:** `Who built the Eiffel Tower?`

**Answer:** `Gustave Eiffel`

### Advanced Configuration
- **Model Switching**: Edit `config.json` to customize settings
- **Database Integration**: Pre-configured for PostgreSQL/MongoDB
- **Custom Training**: Use the training notebook for domain-specific fine-tuning

## 📊 Performance Analysis

Our model demonstrates exceptional performance compared to established baselines:

### Comparison with Research Papers

**Paper 1: "A BERT Baseline for the Natural Questions"**
- **Higher Precision**: Outperforms DocumentQA, DecAtt + DocReader, and BERTjoint
- **Superior Recall**: Better performance than DocumentQA for short answers
- **Best F1 Score**: 80.11% F1 score leads among automated systems for short answers

**Paper 2: "RikiNet: Reading Wikipedia Pages for Natural Question Answering"**
- **Competitive Precision**: 77.50% precision exceeds several baselines including DocumentQA (47.5%), DecAtt + DocReader (52.7%), and BERTjoint (61.3%)
- **Excellent Recall**: 88.57% recall significantly outperforms most automated models
- **Strong F1 Performance**: 80.11% F1 score surpasses many established models

### Model Architecture Details

1. **Dataset Loading**: Uses `cjlovering/natural-questions-short` dataset
2. **Tokenization**: BertTokenizerFast for efficient text processing
3. **Data Preparation**: Converts questions and context into tokenized inputs with answer span labeling
4. **Training**: AdamW optimizer with progress tracking via tqdm
5. **Evaluation**: Comprehensive metrics including Exact Match, F1 score, precision, and recall

## 🎥 Demo

Watch our project demonstration video: [Project Demo Video](https://drive.google.com/file/d/1kjsjkUmLmQhmlRfmuDdyQCDqepdhqAUf/view)

## 👥 Development Team

- **Rahma Ebrahim** - ID: 202201689
- **Bosy Ayman** - ID: 202202076  
- **Jehad Mahmoud** - ID: 202201211

##  Development

### Training Your Own Model
```bash
python src/train_evaluation.py
```

### Running Evaluation
```bash
python src/evaluation.py
```

### Using the Jupyter Notebook
Open `notebooks/NLP_Chatbot_training_&evaluation.ipynb` for interactive development and experimentation.
