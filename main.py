import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# 1. Load your data
# For demo: simulate two product tables
walmart = pd.DataFrame({
    'id': ['w1', 'w2'],
    'name': ['Apple iPhone 12', 'Samsung Galaxy S20']
})

amazon = pd.DataFrame({
    'id': ['a1', 'a2'],
    'name': ['iPhone 12 by Apple', 'Galaxy S20 Samsung Smartphone']
})

# 2. Load model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small and fast

# 3. Compute embeddings
emb_walmart = model.encode(walmart['name'], convert_to_tensor=True)
emb_amazon = model.encode(amazon['name'], convert_to_tensor=True)

# 4. Compute cosine similarity
cos_sim = util.pytorch_cos_sim(emb_walmart, emb_amazon)

# 5. Print most similar matches
for i in range(len(walmart)):
    best_match = torch.argmax(cos_sim[i]).item()
    score = cos_sim[i][best_match].item()
    print(f"{walmart['name'][i]}  <--->  {amazon['name'][best_match]} (score: {score:.2f})")