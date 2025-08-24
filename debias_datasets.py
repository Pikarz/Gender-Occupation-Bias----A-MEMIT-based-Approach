
import re
import pandas as pd 

import re

def _same_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[0].isupper():
        return repl.capitalize()
    return repl.lower()

def swap_genders(text: str) -> str:
    # Base mapping
    base_map = {
        "she": "he",
        "her": "him",      
        "he": "she",
        "him": "her",
    }

    # Build a single regex that matches any pronoun as a whole word (case-insensitive)
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, base_map.keys())) + r")\b", flags=re.IGNORECASE)

    def _repl(m: re.Match) -> str:
        src = m.group(0)
        tgt = base_map[src.lower()]
        return _same_case(src, tgt)

    return pattern.sub(_repl, text)

def clean_and_extract(sentence, mask='<BLANK>'):
    y = re.findall(r"\[(.*?)\]", sentence)  # find all occurrences of [...]
    referent = y[0] # The first occurrence is always the referent
    pronouns = y[1:] # All the others are pronouns
    
    # Remove all brackets -- clean sentence
    anti_stereotyped_sentence = re.sub(r"[\[\]]", "", sentence)

    # Build masked sentence starting from the original sentence
    masked_sentence = sentence
    for p in pronouns:
        masked_sentence = masked_sentence.replace(f"[{p}]", mask)  # mask each pronoun
        masked_sentence = masked_sentence.replace(f"[{referent}]", referent) # remove brackets from the referent

    # Also produce a stereotyped sentence from the clean, anti-stereotyped sentence
    stereotyped_sentence = swap_genders(anti_stereotyped_sentence)

    return anti_stereotyped_sentence, stereotyped_sentence, masked_sentence, referent, pronouns

def load_winobias(path):

    anti_stereotyped_sentences, stereotyped_sentences, masked_sentences = [], [], []
    referents, pronouns = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()[1:]
            sentence = ' '.join(line) # sentence with referent and pronouns in brackets [...]

            anti_stereotyped_sentence, stereotyped_sentence, masked_sentence, referent, pronoun = clean_and_extract(sentence) 

            anti_stereotyped_sentences.append(anti_stereotyped_sentence)
            stereotyped_sentences.append(stereotyped_sentence)
            masked_sentences.append(masked_sentence)
            referents.append(referent)
            pronouns.append(pronoun)           

    return anti_stereotyped_sentences, stereotyped_sentences, masked_sentences, referents, pronouns

def load_winobias_dataframe(path="./dataset/winobias/WinoBias/wino/data/anti_stereotyped_type1.txt.dev"):
    anti_stereotyped_sentences, stereotyped_sentences, masked_sentences, referents, pronouns = load_winobias(path)

    df = pd.DataFrame({ 'anti_stereotyped_sentence': anti_stereotyped_sentences, 
                        'stereotyped_sentence' : stereotyped_sentences,
                        'masked_sentence': masked_sentences, 
                        'referent': referents, 
                        'pronouns': pronouns, 
                     })
    return df

if __name__ == '__main__':
    wino_df = load_winobias_dataframe()

    for _, data in wino_df.iterrows():
        print(data)
        for att in wino_df.columns:
            print(att, data[att])
            
        break


   # data_path_anti = "./dataset/winobias/WinoBias/wino/data/anti_stereotyped_type1.txt.dev"
   # clean_anti, labels_anti = load_winobias(data_path_anti)