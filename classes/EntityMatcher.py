import pandas as pd
import torch
import os
import re
import sys
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

# Add parent directory to path to import locallib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from locallib import *

class EntityMatcher:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('all-mpnet-base-v2', device=self.device)
        self.threshold = 0.93

    def make_text(self, df):
        return (
            df['title'].fillna('') + ' ' +
            df['brand'].fillna('') + ' ' +
            df['category'].fillna('') + ' ' +
            df['shortdescr'].fillna('') + ' ' +
            df['techdetails'].fillna('')
        ).str.lower()

    def clean(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9 ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def entity_match_amazon_walmart(self):
        # Check if the file is available
        if not os.path.exists('amazon_walmart.tsv'):
            print("File 'amazon_walmart.tsv' not found. Please ensure the file is in the correct directory.")
            return None
        
        
        # Read the TSV file into a DataFrame    
        df = pd.read_csv('amazon_walmart.tsv', sep='\t')
        df = df.dropna(subset=['brand', 'title'])
        df['brand'] = df['brand'].str.lower() # Normalize brand names (lowercase for consistency)

        # Separate amazon products and walmart products
        amazon = df[df['dataset'] == 'amazon'].reset_index(drop=True)
        walmart = df[df['dataset'] == 'walmart'].reset_index(drop=True)

        # Make groups for blocking
        grouped_amazon = amazon.groupby('brand')
        grouped_walmart = walmart.groupby('brand')

        # Shared brands only
        shared_brands = set(grouped_amazon.groups).intersection(grouped_walmart.groups)

        matches = []

        # Loop through each shared brand.
        for brand in shared_brands:
            amazon_group = grouped_amazon.get_group(brand)    # creates dataframe for a brand, so that only them will be searched for matching
            walmart_group = grouped_walmart.get_group(brand) 

            # Create textual representations of the products.
            amazon_text =  self.make_text(amazon_group).apply(self.clean).tolist()
            walmart_text = self.make_text(walmart_group).apply(self.clean).tolist()

            # Do not track grading to free memory.
            with torch.no_grad():
                # Embed the textual representations to a vector representation
                embedded_amazon = self.model.encode(amazon_text, convert_to_tensor=True, device=self.device, batch_size=64)
                embedded_walmart = self.model.encode(walmart_text, convert_to_tensor=True, device=self.device, batch_size=64)

                # Compute cosine similarity between the two sets of embeddings
                # Cosine similarity calculates the cosine of the angle between two vectors
                # cosine = 1 means the vectors are identical, cosine = 0 means they are orthogonal
                # this is a 2D-array with similarities of each amazon product with walmart product
                cosine_similarities = util.pytorch_cos_sim(embedded_amazon, embedded_walmart)

            ## now find the matches.
            # Batch find best match indices and scores for all Amazon products
            best_scores, best_indices = torch.max(cosine_similarities, dim=1)  # dim=1 means "per row"
           
            # Preload rows to avoid iloc overhead
            amazon_rows = amazon_group.reset_index(drop=True)
            walmart_rows = walmart_group.reset_index(drop=True)

            for i in torch.where(best_scores >= self.threshold)[0]:  # i is a tensor index of accepted matches
                i = i.item()
                j = best_indices[i].item()
                matches.append({
                    'amazon_id': amazon_rows.iloc[i]['id'],
                    'amazon_title': amazon_rows.iloc[i]['title'],
                    'walmart_id': walmart_rows.iloc[j]['id'],
                    'walmart_title': walmart_rows.iloc[j]['title'],
                    'similarity': best_scores[i].item(),
                    'brand': brand
                })


        # Convert to DataFrame and view
        return pd.DataFrame(matches)


    def alternative_matching(self):
        # Check if the file is available
        if not os.path.exists('amazon_walmart.tsv'):
            print("File 'amazon_walmart.tsv' not found. Please ensure the file is in the correct directory.")
            return None
        
        
        # Read the TSV file into a DataFrame    
        df = pd.read_csv('amazon_walmart.tsv', sep='\t')
        df = df.dropna(subset=['brand', 'title'])
        df['brand'] = df['brand'].str.lower() # Normalize brand names (lowercase for consistency)

        # Separate amazon products and walmart products
        amazon = df[df['dataset'] == 'amazon'].reset_index(drop=True)
        walmart = df[df['dataset'] == 'walmart'].reset_index(drop=True)

        # Make groups for blocking
        grouped_amazon = amazon.groupby('brand')
        grouped_walmart = walmart.groupby('brand')

        # Shared brands only
        shared_brands = set(grouped_amazon.groups).intersection(grouped_walmart.groups)

        matches = []

        # Loop through each shared brand.
        for brand in shared_brands:
            amazon_group = grouped_amazon.get_group(brand)    # creates dataframe for a brand, so that only them will be searched for matching
            walmart_group = grouped_walmart.get_group(brand) 

            # Create textual representations of the products.
            amazon_text =  self.make_text(amazon_group).apply(self.clean).tolist()
            walmart_text = self.make_text(walmart_group).apply(self.clean).tolist()

            # Encode Walmart texts
            walmart_emb = self.model.encode(walmart_text, convert_to_numpy=True, normalize_embeddings=True)
            amazon_emb = self.model.encode(amazon_text, convert_to_numpy=True, normalize_embeddings=True)

            # Convert to float32 (FAISS needs this)
            walmart_emb = np.asarray(walmart_emb, dtype='float32')
            amazon_emb = np.asarray(amazon_emb, dtype='float32')

            # Build FAISS index (cosine similarity via inner product on normalized vectors)
            index = faiss.IndexFlatIP(walmart_emb.shape[1])  # IP = inner product
            index.add(walmart_emb)  # Add Walmart vectors

            # Search: find best match for each Amazon vector
            # k = 1 for top-1 match
            scores, indices = index.search(amazon_emb, k=1)  # shapes: [n_amazon, 1]

            # Build matches
            for i in range(len(amazon_emb)):
                score = scores[i][0]
                j = indices[i][0]
                
                if score >= self.threshold:
                    matches.append({
                        'amazon_id': amazon_group.iloc[i]['id'],
                        'amazon_title': amazon_group.iloc[i]['title'],
                        'walmart_id': walmart_group.iloc[j]['id'],
                        'walmart_title': walmart_group.iloc[j]['title'],
                        'similarity': float(score),
                        'brand': brand
                    })

            return pd.DataFrame(matches)