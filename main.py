import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

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

# Define the ai model.
model = SentenceTransformer('all-MiniLM-L6-v2')

matches = []

# Loop through each shared brand.
for brand in shared_brands:
    amazon_group = grouped_amazon.get_group(brand)    # creates dataframe for a brand, so that only them will be searched for matching
    walmart_group = grouped_walmart.get_group(brand) 

    # Create textual representations of the products.
    amazon_text = (amazon_group['title'] + ' ' + amazon_group['brand']).tolist()
    walmart_text = (walmart_group['title'] + ' ' + walmart_group['brand']).tolist()

    # Embed the textual representations to a vector representation
    embedded_amazon = model.encode(amazon_text, convert_to_tensor=True)
    embedded_walmart = model.encode(walmart_text, convert_to_tensor=True)

    # Compute cosine similarity between the two sets of embeddings
    # Cosine similarity calculates the cosine of the angle between two vectors
    # cosine = 1 means the vectors are identical, cosine = 0 means they are orthogonal
    cosine_similarity = util.pytorch_cos_sim(embedded_amazon, embedded_walmart)

    ## now find the matches.

    for i in range(len(amazon_group)):
        best_idx = torch.argmax(cosine_similarity[i]).item()
        best_score = cosine_similarity[i][best_idx].item()
        matches.append({
            'amazon_id': amazon_group.iloc[i]['id'],
            'amazon_title': amazon_group.iloc[i]['title'],
            'walmart_id': walmart_group.iloc[best_idx]['id'],
            'walmart_title': walmart_group.iloc[best_idx]['title'],
            'similarity': best_score,
            'brand': brand
        })


# Convert to DataFrame and view
matched_df = pd.DataFrame(matches)

# Write matches to a file.
matched_df.to_csv("matches.csv", index=False)
print(matched_df.head(10))