def truncate(text, maxlength=50):
    text = str(text)
    if (len(text) <= maxlength) :
        return text
    else :
        return text[:maxlength - 3] + "..."
