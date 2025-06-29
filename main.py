from tabulate import tabulate
from sentence_transformers import SentenceTransformer, util
from locallib import *
from classes.EntityMatcher import EntityMatcher
from classes.GeneralEntityMatcher import GeneralEntityMatcher
import streamlit as st

# Get the matched dataframe of amazon and walmart products.
#em = EntityMatcher()
gem = GeneralEntityMatcher(0.95)

#result_table = gem.match_entities('booking_hotels.csv', 'expedia_hotels.csv')
result_table = gem.match_entities('amazon_products.csv', 'walmart_products.csv')
#Pretty-print truncated table
with open("matches.txt", "w", encoding="utf-8") as f:
    f.write(tabulate(result_table, headers="keys", tablefmt="fancy_grid"))

st.title("Entitymatching")
st.dataframe(result_table)

#print(result_table.head(10))