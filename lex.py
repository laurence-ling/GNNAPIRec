import re
import numpy as np
from gensim.models import Word2Vec


class LexParser:
    def __init__(self, sents, dim=64):
        self.sents = sents
        self.dim = dim
        self.model = self.__pretrain()
        self.model.save('word2vec.pretrain')
        self.pre_embedding = [np.zeros(self.dim, dtype=np.float32)]
        self.vocab = self.build_vocab_dict()
        print('vocab size:', len(self.vocab))

    def build_vocab_dict(self):
        index = 1
        d = {}
        for key in self.model.wv.vocab:
            d[key] = index
            self.pre_embedding.append(self.model[key])
            index += 1
        return d

    def vectorize(self, name):
        tokens = self.parse_signature(name)
        return [self.vocab[w] for w in tokens]

    def __pretrain(self):
        self.sents = [self.parse_signature(s) for s in self.sents]
        print('sents size:', len(self.sents))
        return Word2Vec(self.sents, size=self.dim, min_count=1, workers=4, sg=1)

    def get_embedding(self, sig):
        words = self.parse_signature(sig)
        embedding = np.zeros(self.dim, dtype=np.float32)
        try:
            if words:
                embedding = sum([self.model[w] for w in words]) / len(words)
        except KeyError as e:
            print('word not in vocab', sig, e)
        return embedding

    @staticmethod
    def parse_signature(s):
        s = s.split('(')[0]
        s = s.split('/')
        ret = []
        for name in s:
            words = LexParser.camel_case(name)
            for w in words:
                w = LexParser.snake_case(w)
                if w:
                    ret += w
        return ret

    @staticmethod
    def camel_case(name):
        words = re.findall('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', name)
        return [w.lower() for w in words]

    @staticmethod
    def snake_case(name):
        res = []
        for w in name.split('_'):
            if re.match(r'\d+', w):
                continue
            w = LexParser.filter_number(w)
            if w and len(w) > 1:
                res.append(w)
        return res

    @staticmethod
    def filter_number(name):
        pattern = re.compile(r'(\w*[a-z])(\d*$)')
        match = pattern.match(name)
        if not match:
            return None
        return pattern.match(name).group(1)


if __name__ == '__main__':
    #parser = LexParser(['Abc/BdbHi/jkLm(int i)', 'lex/LexParser/parse(string)'])
    #embedding = parser.get_embedding('abc/parse/ijk')
    model = Word2Vec.load('word2vec.pretrain')
    print(model.wv.vectors.shape, model.wv.vocab.keys())
