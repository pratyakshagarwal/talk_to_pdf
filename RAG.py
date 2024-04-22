import fitz
import torch
import requests
import regex as re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from spacy.lang.en import English
from sentence_transformers import util, SentenceTransformer

def text_formatter(text: str) -> str:
    ''' Perform minor formatting on text.
    
    Args:
        text (str): The input text to be cleaned.
        
    Returns:
        str: The cleaned text with removed newlines and trailing spaces.
    '''
    cleaned_text = text.replace('\n', ' ').strip()

    # Potentially more text formatting functions can go here

    return cleaned_text


import textwrap

def print_wrapped(text, wrap_length=80):
    ''' Print text with wrapping to a specified line length.
    
    Args:
        text: The text to be printed.
        wrap_length (int): The maximum length for each line.
    '''
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


class RAG:
    def __init__(self) -> None:
        self.pages_and_text = list()
        self.pages_and_chunks = list()
        self.pages_and_chunks_over_minlen = list()
        self.embedding_model = None
        self.embeddings = []

    def retrieve_relevant_resources(self, query: str, k: int=5):
        ''' Retrieve relevant resources based on a query.
        
        Args:
            query (str): The query string.
            k (int): Number of top results to return.
            
        Returns:
            torch.Tensor: Scores of the top-k results.
            torch.Tensor: Indices of the top-k results.
        '''
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        dot_scores = util.dot_score(query_embedding, self.embeddings)[0]
        scores, indices = torch.topk(input=dot_scores, k=k)
        return scores, indices
    
    def print_top_results(self, query: str, k: int=5):
        ''' Print the top results for a given query.
        
        Args:
            query (str): The query string.
            k (int): Number of top results to print.
        '''
        scores, indices = self.retrieve_relevant_resources(query=query, k=k)
        print(f'Query: {query}\n')
        print('Results:')
        for score, index in zip(scores, indices):
            print(f"Score: {score:.4f}")
            print_wrapped(self.pages_and_chunks[index]['sentence_chunk'])
            print(f"Page number: {self.pages_and_chunks[index]['page_number']}\n")

    def train(self, pdf_path: str, pdf_strt_page_no: int=None, pdf_end_page_no: int=None, tokenizer=None, chunk_size: int=10) -> None:
        ''' Train the RAG model on a PDF document.
        
        Args:
            pdf_path (str): Path to the PDF file.
            pdf_strt_page_no (int): Starting page number (default: None).
            pdf_end_page_no (int): Ending page number (default: None).
            tokenizer: Optional tokenizer.
            chunk_size (int): Size of sentence chunks.
        '''
        doc = fitz.open(pdf_path)
        if pdf_strt_page_no is None:
            pdf_strt_page_no = 0
        if pdf_end_page_no is None:
            pdf_end_page_no = len(doc)
        for page_num, page in tqdm(enumerate(doc[pdf_strt_page_no: pdf_end_page_no])):
            text = page.get_text()
            text = text_formatter(text=text)
            self.pages_and_text.append({
                'page_number': page_num,
                'page_char_count': len(text),
                'word_count': len(text.split(' ')),
                'page_token_count': len(text.split()) // 4,
                'text': text
            })

        self.tokenize()

        for item in tqdm(self.pages_and_text):
            item['sentence_chunks'] = self.split_list(input_list=item['sentences'], chunk_size=chunk_size)
            item['num_chunks'] = len(item['sentence_chunks'])

        for item in tqdm(self.pages_and_text):
            for sentence_chunk in item['sentence_chunks']:
                chunk_dict = {}
                chunk_dict['page_number'] = item['page_number']

                joined_sentence_chunk = ''.join(sentence_chunk).replace(' ', ' ').strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
                chunk_dict['sentence_chunk'] = joined_sentence_chunk

                chunk_dict['chunk_char_count'] = len(joined_sentence_chunk)
                chunk_dict['chunk_word_count'] = len(joined_sentence_chunk.split(' '))
                chunk_dict['chunk_token_count'] = len(joined_sentence_chunk) / 4

                self.pages_and_chunks.append(chunk_dict)

    def saturate_min_token(self, minlen=30):
        ''' Filter chunks based on a minimum token length.
        
        Args:
            minlen (int): Minimum token length to filter by.
        '''
        df = pd.DataFrame(self.pages_and_chunks)
        self.pages_and_chunks_over_minlen = df[df['chunk_token_count'] > minlen].to_dict(orient='records')

    def load_embedding_model(self, model_name: str='all-mpnet-base-v2', device: str='cpu') -> None:
        ''' Load the embedding model.
        
        Args:
            model_name (str): Name or path of the embedding model.
            device (str): Device to load the model on.
        '''
        self.embedding_model = SentenceTransformer(model_name_or_path=model_name, 
                                                   device=device).to(device)

    def embed_chunks(self, model_name='all-mpnet-base-v2', device: str='cpu') -> None:
        ''' Embed sentence chunks using the loaded model.
        
        Args:
            model_name (str): Name or path of the embedding model.
            device (str): Device to load the model on.
        '''
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(model_name_or_path=model_name,
                                                       device=device).to(device)
            
        for item in tqdm(self.pages_and_chunks_over_minlen):
            item['embedding'] = self.embedding_model.encode(item['sentence_chunk'])

    def make_embeddings_vector(self, filepath: str='text_chunks_nd_embeddings_df.csv', device: str='cpu'):
        ''' Make embeddings vector from the chunks and save to file.
        
        Args:
            filepath (str): Filepath to save the embeddings.
            device (str): Device to use for torch tensor operations.
        '''
        text_chunks_and_embedding_df = pd.read_csv(filepath)
        # Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
        text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        # Convert texts and embedding df to list of dicts
        self.pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
        # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
        self.embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)

    def save_embedding_to_file(self, filepath: str='text_chunks_nd_embeddings_df.csv', device: str='cpu'):
        ''' Save the embeddings to a file.
        
        Args:
            filepath (str): Filepath to save the embeddings.
            device (str): Device to use for torch tensor operations.
        '''
        # Save embeddings to file
        text_chunks_and_embeddings_df = pd.DataFrame(self.pages_and_chunks_over_minlen)
        text_chunks_and_embeddings_df.to_csv(filepath, index=False)

    def get_stats(self, round: int=2) -> None:
        ''' Print statistics about the chunks.
        
        Args:
            round (int): Number of decimal places to round to.
        '''
        df = pd.DataFrame(self.pages_and_chunks)
        print(df.describe().round(round))

        df = pd.DataFrame(self.pages_and_chunks_over_minlen)
        print(df.describe().round(round))
    
    def tokenize(self, tokenizer=None) -> None:
        ''' Tokenize the text into sentences.
        
        Args:
            tokenizer: Optional tokenizer.
        '''
        if tokenizer is None:
            nlp = English()
            # Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/ 
            nlp.add_pipe("sentencizer")

        for item in tqdm(self.pages_and_text):
            item['sentences'] = list(nlp(item['text']).sents)
            item['sentences'] = [str(sentence) for sentence in item['sentences']]
            item['page_sentence_count_spcay'] = len(item['sentences'])

    def split_list(self, input_list: list, chunk_size: int=10) -> None:
        ''' Split a list into chunks of a specified size.
        
        Args:
            input_list (list): The input list to be split.
            chunk_size (int): Size of each chunk.
        
        Returns:
            list: List of chunks.
        '''
        return [input_list[i:i+chunk_size] for i in range(0, len(input_list), chunk_size)]
    
if __name__ == '__main__':
    pdf_path = r'Deep_Learning.pdf'
    embeding_path = r'text_chunks_and_embeddings_df.csv'
    device = 'cpu'
    query = 'Explain Adam optimizer'

    rag = RAG()
    rag.train(pdf_path=pdf_path, 
          pdf_strt_page_no=12,
            pdf_end_page_no=650,)
    rag.saturate_min_token()
    rag.load_embedding_model()
    rag.make_embeddings_vector(filepath=embeding_path)
    rag.print_top_results(query=query, k=5)