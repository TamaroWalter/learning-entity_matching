from tabulate import tabulate
from sentence_transformers import SentenceTransformer, util
from locallib import *
from classes.EntityMatcher import EntityMatcher

# Get the matched dataframe of amazon and walmart products.
em = EntityMatcher()

result_table = em.entity_match_amazon_walmart()

# Pretty-print truncated table
with open("matches.txt", "w", encoding="utf-8") as f:
    f.write(tabulate(result_table, headers="keys", tablefmt="fancy_grid"))

# Write matches to a file.
#matched_df.to_csv("matches.csv", index=False)

print(result_table.head(10))