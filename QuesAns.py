pip install transformers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

# Check if CUDA is available
if torch.cuda.is_available():
    # If CUDA is available, set the device to CUDA
    device = torch.device('cuda')
    print('CUDA is available! Using GPU for computations.')
else:
    # If CUDA is not available, use the CPU
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU for computations.')

# Load your BERT model and move it to the selected device
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad').to(device)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

dataset=pd.read_csv('LLM-Sample-Input-File.csv')

col_con=['Company Name','Category','Sub Cat','Period']
dataset['concatenated_column'] = dataset[col_con].apply(lambda row: ''.join(row.values.astype(str)), axis=1)

# Function to get the answer from the dataset based on the user's question keywords
def get_answer_from_dataset(question, dataset):
    # Split the user question into keywords
    keywords = question.split(' ')
    
    # Find rows in the dataset where concatenated_column contains maximum keywords
    matching_rows = dataset[dataset['concatenated_column'].str.contains('|'.join(keywords))]
    if not matching_rows.empty:
        # Get the row with the maximum matching keywords and retrieve the answer
        max_match_row = matching_rows.iloc[matching_rows['concatenated_column'].apply(lambda x: sum(keyword in x for keyword in keywords)).idxmax()]
        answer = max_match_row['ValueRandomized ']
        return answer
    else:
        return "Unable to find the answer."

# Continuously accept user input for questions until 'exit' is entered
while True:
    user_question = input("Ask a question (type 'exit' to quit): ")
    if user_question.lower() == 'exit':
        break
    
    # Get the answer for the user's question from the dataset
    ans = get_answer_from_dataset(user_question, dataset)
    numeric_value = int(ans.replace(',', ''))
    answer = "{:,}".format(numeric_value)

    # Display the answer
    print("Question:", user_question)
    print("Answer:", answer)
    print("------------")
