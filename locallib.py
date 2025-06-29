import pandas as pd

def truncate(text, maxlength=50):
    text = str(text)
    if (len(text) <= maxlength) :
        return text
    else :
        return text[:maxlength - 3] + "..."

def prepare_text(df: pd.DataFrame):
    """
    This functions takes a dataframe and puts the text from the different colum in one.
    This is needed to convert it into a vector later.
    * df: pd.DataFrame: a dataframe
    * return list: A list of textes
    """
    df = df.fillna('').astype(str)

    text = pd.Series([''] * len(df), index=df.index)
    for column in df.columns:
        text += df[column] + ' '

    # now clean the text.
    text = text.str.lower()
    text = text.str.replace(r'[^a-z0-9 ]', ' ', regex=True)
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()
    return text.tolist()

def read_csv_or_tsv(filename: str) -> pd.DataFrame:
    """
    Reads a CSV or TSV file and returns a DataFrame.
    * filename: str: The name of the file to read.
    * return df: pd.DataFrame: The DataFrame containing the data from the file.
    """
    if filename.endswith('.csv'):
        return pd.read_csv(filename)
    elif filename.endswith('.tsv'):
        return pd.read_csv(filename, sep='\t')
    else:
        raise ValueError("Unsupported file format. Please use .csv or .tsv.")
    
def summarize_text(text, summarizer, min_tokens=10, max_tokens=20):
    if len(text.split()) < min_tokens:
        return text.strip()

    summary = summarizer(
        text,
        max_length=max_tokens,
        max_new_tokens=20,
        min_length=10,
        do_sample=False,
        truncation=True
    )
    return summary[0]['summary_text'].strip()