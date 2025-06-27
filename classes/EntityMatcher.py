import pandas as pd
import torch
import os
import re
import sys
from sentence_transformers import SentenceTransformer, util

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

            for i in range(len(amazon_group)):
                similarity_row = cosine_similarities[i]
                best_idx = torch.argmax(similarity_row).item() # Get best ID
                best_score = similarity_row[best_idx].item() # Get cell from id

                if (best_score >= self.threshold):
                    amazon_row = amazon_group.iloc[i]
                    walmart_row = walmart_group.iloc[best_idx]
                    matches.append({
                        'amazon_id': amazon_row['id'],
                        'amazon_title': amazon_row['title'],
                        'walmart_id':walmart_row['id'],
                        'walmart_title': walmart_row['title'],
                        'similarity': best_score,
                        'brand': brand
                    })


        # Convert to DataFrame and view
        return pd.DataFrame(matches)
