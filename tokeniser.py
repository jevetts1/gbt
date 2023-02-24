import re
import numpy as np

class Subword_Tokeniser():
    def __init__(self):
        self.tokens = []
        self.standard_tokens = ["<\p>","<\w>","<\s>","<\e>","<OOV>"] #padding, end of word, start of sequence, end of sequence, out of vocabulary
        self.vocab = 0

    def fit_on_text_BPE(self,texts,merges):
        words = []

        for line in texts:
            line = line.split()
            line = list(map(lambda x:x.replace(""," ")[1:] + "<\w>",line))
            line[0] = "<\s> " + line[0]
            line[-1] = line[-1].replace("<\w>","<\e>")

            words += line

        words_counts = np.unique(words,return_counts = True)

        word_dict = {}

        for i in range(len(words_counts[0])):
            word_dict.update({words_counts[0][i]:words_counts[1][i]})

        tokens = []
        tokens.extend(self.standard_tokens)

        for i in range(merges):
            best_pair = self.find_best_pair(word_dict)
            word_dict = self.replace_with_pair(word_dict,best_pair)

        for word in word_dict:
            symbols = word.split()

            for token in symbols:
                if token not in tokens:
                    tokens.append(token)
        
        self.tokens = tokens
        self.vocab = len(self.tokens)

    def replace_with_pair(self,word_dict,best_pair):
        new_word_dict = {}

        for word in word_dict:
            new_word_dict.update({word.replace(best_pair,best_pair.replace(" ","")):word_dict[word]})

        return new_word_dict

    def find_best_pair(self,word_dict):
        pairs = {}

        for word,frequency in word_dict.items():
            symbols = word.split()

            for i in range(len(symbols) - 1):
                pair = symbols[i] + " " + symbols[i + 1]

                current_frequency = pairs.get(pair,0)
                pairs[pair] = current_frequency + frequency

        return max(pairs,key = pairs.get)

    def encode(self,texts):
        encoded_texts = []

        for line in texts:
            try:
                line = line.split()
                line = list(map(lambda x:x.replace(""," ")[1:] + "<\w>",line))
                line[0] = "<\s> " + line[0]
                line[-1] = line[-1].replace("<\w>","<\e>")

            except:
                continue
            
            encoded_line = []

            for word in line:
                complete = False

                while not complete:
                    symbols = word.split()
                    changed = False
                    
                    for i in range(len(symbols) - 1):
                        if symbols[i] + symbols[i + 1] in self.tokens:
                            changed = True
                            word = word.replace(symbols[i] + " " + symbols[i + 1],symbols[i] + symbols[i + 1])

                    if not changed:
                        complete = True
                
                for token in word.split():
                    if token in self.tokens:
                        encoded_line.append(self.tokens.index(token))

                    else:
                        encoded_line.append(self.tokens.index("<OOV>"))
            
            encoded_texts.append(encoded_line)
        
        return encoded_texts

    def decode(self,sequences):
        texts = []

        for sequence in sequences:
            text = ""

            for token in sequence:
                text += self.tokens[token]

            text = text.replace("<\w>"," ")
            
            for t in self.standard_tokens:
                text = text.replace(t,"")
                
            texts.append(text)

        return texts

class Subword_Tokeniser_V2():
    def __init__(self):
        self.tokens = []
        self.vocab = 0

    def fit_on_texts(self,texts):
        for line in texts:
            line_chars = 0

if __name__ == "__main__":
    with open("poems.txt","r",encoding = "mbcs") as file:
        poems = file.read()
        file.close()

    t = Subword_Tokeniser()
    t.fit_on_text_BPE([poems],100)
    encoded = t.encode(["What is your name?","Hi there's a man. What will you do!!!! HAHAHAAH!"])
    print(t.decode(encoded))
