import pandas as pd
import numpy as np
from utils import progress_bar

def get_blogs(num_blogs):
    df = pd.read_csv("blogtext.csv")
    blogs = df["text"][:num_blogs]

    return list(blogs)

def blogs_to_train(tokeniser,texts,max_examples,window_size):
    x = []
    y = []

    print("Encoding text...",end = "\r")
    encoded_texts = tokeniser.encode(texts)

    for i,encoded_text in enumerate(encoded_texts):
        print(progress_bar(len(x)  / max_examples),end = "\r")

        if len(encoded_text) <= window_size:
             continue
        
        for i in range(len(encoded_text) - window_size):
            x.append(encoded_text[i:window_size + i])
            y.append(encoded_text[i + 1:window_size + i + 1])

            if len(x) >= max_examples:
                break

        if len(x) >= max_examples:
                break

    return np.array(x),np.array(y)