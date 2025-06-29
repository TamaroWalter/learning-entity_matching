from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd
import torch
import pprint
import json
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from locallib import *

class GeneralEntityMatcher:
    def __init__(self, threshold):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.aiclient = SentenceTransformer('all-mpnet-base-v2', device=self.device)
        self.threshold = threshold
        self.bookingdf = pd.read_csv("booking_hotels.csv")
        self.summarizer = pipeline("summarization", model="Falconsai/text_summarization")

    def match_entities(self, tablename1, tablename2):
        # make sure both file exists
        if not (os.path.exists(tablename1) and os.path.exists(tablename2)):
            print("File not found. Please ensure the files exists.")
            return None

        # Read the csv/tsv file
        # Check if the file is a csv or tsv
        df_one = read_csv_or_tsv(tablename1)
        df_two = read_csv_or_tsv(tablename2)

        # Create text vectors
        df_one_texts = prepare_text(df_one)
        df_two_texts = prepare_text(df_two)
        
        # Create embeddings.
        with torch.no_grad():
            embedded_df_one = self.aiclient.encode(df_one_texts,  convert_to_tensor=True, device=self.device, batch_size=64)
            embedded_df_two = self.aiclient.encode(df_two_texts,  convert_to_tensor=True, device=self.device, batch_size=64)
            cosine_similarities = self.aiclient.similarity(embedded_df_one, embedded_df_two)
        
        best_scores, best_indices = torch.max(cosine_similarities, dim=1)  # dim=1 means "per row"

        # Find best matches and return a dataframe with the following structure:
        # index | summary of content in table1_row | summary of content in table2_row | score
        # where table1_row and table2_row are the full rows of the matched items.
        matches = []
        for idx in torch.where(best_scores >= self.threshold)[0]:
            i = idx.item()
            j = best_indices[i].item()
            matches.append({
            'index': i,
            'table1_row': summarize_text(df_one_texts[i], self.summarizer),
            'table2_row': summarize_text(df_two_texts[j], self.summarizer),
            'score': best_scores[i].item()
            })

        return pd.DataFrame(matches)