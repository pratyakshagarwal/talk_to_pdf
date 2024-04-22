# RAG (Retrieval-Augmented Generation) Model

This repository contains code for a Retrieval-Augmented Generation (RAG) model implemented in Python. The RAG model is designed to retrieve relevant resources from a PDF document based on a query and present the top results.

## Contents

- `rag.py`: Python script containing the RAG class and its methods.
- `Deep_Learning.pdf`: Sample PDF used for demonstration.
- `text_chunks_and_embeddings_df.csv`: CSV file to store the text chunks and their embeddings.
- `README.md`: This README file.

## Dependencies

- `fitz`: PyMuPDF library for PDF processing.
- `torch`: PyTorch library for machine learning.
- `requests`: Library for making HTTP requests.
- `regex`: Library for regular expressions.
- `numpy`: Library for numerical operations.
- `pandas`: Library for data manipulation and analysis.
- `tqdm`: Library for progress bars.
- `spacy`: Library for NLP tasks.
- `sentence_transformers`: Library for sentence embeddings.

## Usage

### Setup

1. Ensure you have the required libraries installed. If not, install them using:

   ```bash
   pip install -r requirements.txt

2. Download the `Deep_Learning.pdf` file or replace it with your PDF file.
3. Ensure `text_chunks_and_embeddings_df.csv` is available for storing embeddings.

### Running the RAG Model
Modify the following variables in rag.py as needed:

```bash
pdf_path = 'Deep_Learning.pdf'
pdf_strt_page_no = 12
pdf_end_page_no = 650
embeding_path = 'text_chunks_and_embeddings_df.csv'
device = 'cpu'
query = 'Explain Adam optimizer'

Run rag.py 
```
to train the RAG model, load embeddings, and retrieve top results for a query:


### RAG Class Methods
- retrieve_relevant_resources(query, k): Retrieve relevant resources based on a query.
- print_top_results(query, k): Print the top results for a given query.
- train(pdf_path, pdf_strt_page_no, pdf_end_page_no, tokenizer, chunk_size): Train the RAG model on a PDF document.
- saturate_min_token(minlen): Filter chunks based on a minimum token length.
- load_embedding_model(model_name, device): Load the embedding model.
- embed_chunks(model_name, device): Embed sentence chunks using the loaded model.
- make_embeddings_vector(filepath, device): Make embeddings vector from the chunks and save to file.
- save_embedding_to_file(filepath, device): Save the embeddings to a file.
- get_stats(round): Print statistics about the chunks.
- tokenize(tokenizer): Tokenize the text into sentences.
- split_list(input_list, chunk_size): Split a list into chunks of a specified size.


### Example Usage
```bash
# Create an instance of the RAG model
rag = RAG()

# Train the RAG model on the specified PDF document
rag.train(pdf_path='Deep_Learning.pdf', pdf_strt_page_no=12, pdf_end_page_no=650)

# Filter chunks based on a minimum token length
rag.saturate_min_token()

# Load the embedding model
rag.load_embedding_model()

# Make embeddings vector from the chunks and save to file
rag.make_embeddings_vector(filepath='text_chunks_and_embeddings_df.csv')

# Print the top results for the query "Explain Adam optimizer"
rag.print_top_results(query='Explain Adam optimizer', k=5)
```

### vTest Cases
#### Test Case 1: Query for "Explain Adam optimizer"
Expected Output:

```bash
Query: Explain Adam optimizer

Results:
Score: 0.6291
8.4.3 Adam Adam (Kingma and Ba, 2014) is yet another adaptive learning rate
optimization algorithm and is presented in Algorithm 8.7. In the context of the
earlier algo- rithms, it is perhaps best seen as a variant on RMSprop+momentum
with a few important distinctions. First, in Adam, momentum is incorporated
directly as an estimate of the ﬁrst order moment (with exponential weighting) of
the gradient. The most straightforward way to add momentum to RMSprop is to
apply momen- tum to the rescaled gradients which is not particularly well
motivated. Second, Adam includes bias corrections to the estimates of both the
ﬁrst-order moments (the momentum term) and the (uncentered) second order moments
to account for their initialization at the origin (see Algorithm 8.7). RMSprop
also incor- porates an estimate of the (uncentered) second order moment, however
it lacks the correction term. Thus, unlike in Adam, the RMSprop second-order
moment.
Page number: 254
```

### Conclusion
This RAG model is designed to provide a method for retrieving relevant resources from a PDF document based on user queries. It uses sentence embeddings to match queries with chunks of text from the PDF. The README provides an overview of the model, usage instructions, and an example test case.

```bash

This README.md provides a structured overview of the RAG model, including setup instructions, usage examples, and a test case. It also includes properly formatted code blocks with the appropriate language for syntax highlighting.
```
