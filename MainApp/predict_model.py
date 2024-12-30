#Testing data with the trained AI model

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import re

# Function to preprocess the text (same as during training)
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.lower()

# Path to the saved BERT model and tokenizer
model_path = '/Users/jiahui/helpdesk/MainApp/bert_issue_classifier'

# Load the trained BERT model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.to(device)


# Actual label to index mapping from your dataset
label_to_index = {
    'Enrollment': 0,
    'Signature': 1,
    'Scholarship': 2,
    'Transcript': 3,
    'Examination': 4,
    'Other': 5
}

index_to_label = {0: 'Enrollment', 1: 'Signature', 2: 'Scholarship', 3: 'Transcript', 4: 'Examination', 5: 'Other'}

# Function to predict issue type from the text
def predict_issue_type(text, model, tokenizer, max_len=128):
    # Preprocess the input text
    text = preprocess_text(text)

    # Tokenize the input text
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Get model predictions
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        inputs = {key: val.to(model.device) for key, val in encoding.items()}
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted label (index of the highest logit)
    predicted_label_idx = np.argmax(logits.detach().cpu().numpy(), axis=1)[0]

    return predicted_label_idx

# Function to map label index back to the label name
def map_label_to_name(label_idx, index_to_label):
    return index_to_label[label_idx]

# Sample test data
test_data = [
    "Hey, I need someone to sign my form for the exchange program,",
    "Do you have info on any scholarships I can apply for",
    "I still haven’t received my transcript, any updates,",
    "If I want to start a club, where do I even begin,",
    "I’m thinking about applying for a postgrad program, what should I get ready,",
    "I lost my transcript, can you email it to me,",
    "I can't register for this class, can you help me out,",
    "Can I get a signature for my exchange program,",
    "What do I need to do if I want to start a society,",
    "Any info on scholarships available,",
    "I’ve lost my transcript, can you send me a copy,",
    "I’m interested in postgrad, what should I prepare,",
    "If I want to change my course, is that possible, and how do I do it,",
    "Eh, saya perlukan orang untuk tandatangan borang program pertukaran saya,",
    "Ada tak maklumat tentang biasiswa yang boleh saya mohon,",
    "Saya masih belum terima transkrip saya, ada apa-apa perkembangan,",
    "Kalau saya nak mula persatuan, macam mana nak mula,",
    "Saya tengah fikir nak mohon program pascasiswazah, apa yang perlu saya sediakan,",
    "Saya dah hilang transkrip, boleh tak emailkan kepada saya,",
    "Saya tak dapat daftar kelas ni, boleh tolong saya,",
    "Boleh tak saya dapat tandatangan untuk program pertukaran saya,",
    "Apa yang perlu saya buat kalau saya nak mula persatuan,",
    "Ada tak maklumat tentang biasiswa yang boleh saya mohon,",
    "Saya dah hilang transkrip, boleh tak hantar salinan kepada saya,",
    "Saya berminat dengan program pascasiswazah, apa yang perlu saya sediakan,",
    "Kalau saya nak tukar kursus, boleh ke, dan macam mana saya nak buat,",
    "When is the exam"
]

# Predict the labels for the test data
for text in test_data:
    label_idx = predict_issue_type(text, model, tokenizer)
    label_name = map_label_to_name(label_idx, index_to_label)
    print(f"Text: {text}\nPredicted Issue Type: {label_name}\n")
