import tiktoken
from collections import  Counter
encoding = tiktoken.get_encoding("cl100k_base")

#initially the corpus or raw_text is declared empty
corpus = ""

#opening the file and wrting the contents to the corpus
# with open('./TrainingData/the-verdict.txt','r') as f:
#     corpus = f.read()


corpus = "hello whoa is thisa "

#this will add "</w>" at end of each word  as this will help our llm to understand the last letters or boundary (last char with '</w>' token )
def add_word_boundary(corpus):
    words = corpus.split()
    words_with_boundary = [word + '</w>' for word in words]
    vocab = ' '.join(words_with_boundary)
    return vocab

# converts each word into unique tokens 
def get_bpeTokens(corpus):
    tokens = []
    for word in corpus.split():
        tokens.extend(encoding.encode(word))
    return tokens

# to get the back the actual word using tokens
def get_actualChars(tokens):
    decoded_char = ""
    for token in tokens:
        decoded_char += encoding.decode([token]) + " "
    return decoded_char

#getting the tokens and words and assingining it to the var
tokens = get_bpeTokens(corpus)
words = get_actualChars(tokens)


# prints each subword with its token and decoded chars ( token -> subword ) ONLY FOR LEARNING
def get_understanding(tokens):
    for token in tokens:
        decoded = encoding.decode([token])
        print(f"{token} -> {decoded}")


print("number of tokens:",len(tokens))
print(tokens)
print(words)