# BERT Question Answering System

This code provides a Python implementation of a Question Answering system using BERT (Bidirectional Encoder Representations from Transformers). The system takes a dataset containing information about companies and their financial details and allows users to ask questions to retrieve specific pieces of information from this dataset.

## Requirements
- `transformers` library (`pip install transformers`)
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `torch`

## Usage
1. **Installation:** Install necessary libraries by running `pip install transformers`.
2. **Dataset:** Ensure the dataset (`LLM-Sample-Input-File.csv`) is placed in the same directory as the script.
3. **Running the Code:** Execute the provided Python script.
4. **Interaction:** Enter your questions to retrieve information from the dataset. Type 'exit' to end the session.

## How it Works
- The code utilizes BERT-based models for Question Answering.
- It loads a pretrained BERT model (`bert-large-uncased-whole-word-masking-finetuned-squad`) and tokenizer.
- The dataset (`LLM-Sample-Input-File.csv`) is loaded and a concatenated column is created for efficient searching.
- User questions are processed to extract relevant keywords.
- Based on these keywords, the code searches the dataset for matching rows and retrieves information.

## Important Notes
- **CUDA Usage:** The code checks for GPU availability and utilizes it for computations if possible.
- **Unused Model Weights:** Some weights from the BERT model might not be used due to differences in model tasks or architectures. This behavior is explained in the displayed message during initialization.

Feel free to customize the code or integrate it into your projects as needed! If you encounter any issues or have suggestions, please let us know.

---

*Remember:* This system is designed to retrieve specific information from the dataset based on user questions. Modify the dataset or extend functionalities to suit your requirements.
