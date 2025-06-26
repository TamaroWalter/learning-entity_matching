import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Read the TSV file into a DataFrame
df = pd.read_csv('amazon_walmart.tsv', sep='\t')

# Separate amazon products and walmart products
amazon = df[df['dataset'] == 'amazon'].reset_index(drop=True)
walmart = df[df['dataset'] == 'walmart'].reset_index(drop=True)


# Define the ai model.
model = SentenceTransformer('all-MiniLM-L6-v2')


# Create textual representations of the products
# fillna removes NaN values
amazon_text = (amazon['title'].fillna('') + ' ' + amazon['brand'].fillna('')).tolist()
walmart_text = (walmart['title'].fillna('') + ' ' + walmart['brand'].fillna('')).tolist()

# Embed the textual representations to a vector representation
embedded_amazon = model.encode(amazon_text, convert_to_tensor=True)
embedded_walmart = model.encode(walmart_text, convert_to_tensor=True)

# Compute cosine similarity between the two sets of embeddings
# Cosine similarity calculates the cosine of the angle between two vectors
# cosine = 1 means the vectors are identical, cosine = 0 means they are orthogonal
cosine_similarity = util.pytorch_cos_sim(embedded_amazon, embedded_walmart)


# Now i want to make the matches.
matches = []
for i in range(len(amazon)):
    best_idx = torch.argmax(cosine_similarity[i]).item()
    best_score = cosine_similarity[i][best_idx].item()

    matches.append({
        'amazon_id': amazon.iloc[i]['id'],
        'amazon_title': amazon.iloc[i]['title'],
        'walmart_id': walmart.iloc[best_idx]['id'],
        'walmart_title': walmart.iloc[best_idx]['title'],
        'similarity': best_score
    })

# Convert to DataFrame and view
matched_df = pd.DataFrame(matches)
print(matched_df.head(10))