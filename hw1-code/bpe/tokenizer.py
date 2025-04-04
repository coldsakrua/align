import re
import json
from collections import defaultdict
from tqdm import tqdm


class Tokenizer:
    def __init__(self):
        self.vocab = {}  # str2int
        self.reverse_vocab = {}  # int2str

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for spaced_text, freq in vocab.items():
            tokens = spaced_text.split()
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        new_token = ''.join(pair)

        for spaced_text, freq in vocab.items():
            new_spaced_text = pattern.sub(new_token, spaced_text)
            v_out[new_spaced_text] = freq
        return v_out

    def train(self, text, vocab_size):
        spaced_text = ' '.join(list(text))
        vocab = defaultdict(int)
        vocab[spaced_text] = 1
        unique_chars = set(text)

        self.vocab = {}
        for i, c in enumerate(sorted(unique_chars)):
            self.vocab[c] = i

        current_size = len(self.vocab)
        merges_needed = vocab_size - current_size
        merges_needed = max(0, merges_needed)
        merges_needed = 500
        for _ in tqdm(range(merges_needed)):
            pairs = self.get_stats(vocab)
            # print(pairs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best_pair, vocab)
            new_token = ''.join(best_pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        if "[UNK]" not in self.vocab:
            unk_id = len(self.vocab)
            self.vocab["[UNK]"] = unk_id
            self.reverse_vocab[unk_id] = "[UNK]"

    def encode(self, text):
        chars = list(text)
        result_tokens = []
        i = 0
        while i < len(chars):
            best_match = chars[i] if chars[i] in self.vocab else "[UNK]"

            j = i + 1
            while j <= len(chars):
                candidate = ''.join(chars[i:j])
                if candidate in self.vocab:
                    best_match = candidate
                    j += 1
                else:
                    break

            result_tokens.append(best_match)
            if best_match == "[UNK]":
                i += 1
            else:
                i += len(best_match)

        ids = [self.vocab.get(t, self.vocab["[UNK]"]) for t in result_tokens]
        return ids

    def decode(self, ids):
        tokens = [self.reverse_vocab[i] for i in ids]
        return ''.join(tokens)

    def save(self, vocab_path='vocab.json', tokenizer_path='tokenizer.json'):
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)

        tokenizer_config = {
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": self.vocab
            },
        }
        with open(tokenizer_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, vocab_path='vocab.json', tokenizer_path='tokenizer.json'):
        tokenizer = cls()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            tokenizer.vocab = json.load(f)
        tokenizer.reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        return tokenizer

    
if __name__=="__main__":
    tokenizer=Tokenizer()
    vocab_size=1024
    with open('./manual.txt',encoding='utf-8') as f:
        text=f.read()
        tokenizer.train(text,vocab_size=vocab_size)
        encoded=tokenizer.encode(text)
        
        # print("original text:",text)
        decoded=tokenizer.decode(encoded)
        
        # print("decoded text:",decoded)
        
        
        tokenizer.save()
        assert(decoded==text)