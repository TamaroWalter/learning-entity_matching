from tabulate import tabulate
from sentence_transformers import SentenceTransformer, util
from locallib import *
from classes.EntityMatcher import EntityMatcher
import streamlit as st

# Get the matched dataframe of amazon and walmart products.
em = EntityMatcher()

result_table = em.entity_match_amazon_walmart()

# Pretty-print truncated table
with open("matches.txt", "w", encoding="utf-8") as f:
    f.write(tabulate(result_table, headers="keys", tablefmt="fancy_grid"))

st.title("Amazon ↔ Walmart Entity Matches")
st.dataframe(result_table)

print(result_table.head(10))