# -*- coding: utf-8 -*-

import spacy
from tqdm import tqdm
from spacy.pipeline import Pipe
import numpy as np
from scipy.sparse import csr_matrix, diags
from multiprocessing import Pool, cpu_count
import itertools
from collections import Counter
cores = cpu_count()

def get_vocab(tokenized_corpus):
    """Get Vocabulary
    This function creates a vocabulary from a tokenized corpus.
    
    
    Args:
        tokenized_corpus (:obj:`list` of :obj:`list`): The tokenized corpus.

    Returns:
       :obj:`list` of :obj:`tupple`: the created vocabulary with the words counts.

    """
    words = list(itertools.chain.from_iterable(tokenized_corpus))
    return Counter(words).most_common()

def index_corpus(tokenized_corpus, vocab):
    """Corpus Indexer
    This function indexes the given tokenized corpus from its vocabulary.
    
    
    Args:
       tokenized_corpus (:obj:`list` of :obj:`list`): The tokenized corpus (list of lists).
       vocab (:obj:`list` of :obj:`tupple`): The vocabulary of the corpus.

    Returns:
       :obj:`list` of :obj:`list`: the indexed corpus.

    """
    vocab_indexer = dict([(word[0], i) for i, word in enumerate(vocab)])
    for i, sentence in enumerate(tokenized_corpus):
        tokenized_corpus[i] = [vocab_indexer[word] for word in sentence]
    return tokenized_corpus

def to_BoW(indexed_corpus, vocabulary):
    """Bag of Word creator
    This function creates a sparse matrix of Bag of Words from an indexed 
    corpus and its vocabulary.
    
    
    Args:
       indexed_corpus (:obj:`list` of :obj:`list`): The indexed corpus (list of lists).
       vocab (:obj:`list` of :obj:`tupple`): The vocabulary of the corpus.

    Returns:
       sparse matrix: the Bags of Words.

    """
    indptr = [0]
    indices = []
    data = []
    for d in indexed_corpus:
        for index in d:
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
    return csr_matrix((data, indices, indptr), dtype=int)

def get_idf(BoW_corpus): 
    """Get idf
    This function calcules the Inverse Document Frequency of the vocabulary.
    
    
    Args:
       BoW_corpus (sparse matrix): The tokenized corpus (list of lists).

    Returns:
       ndarray: the inverse document frequency.

    """
    n_samples, n_features = BoW_corpus.shape
    
    df = np.bincount(BoW_corpus.indices, minlength=BoW_corpus.shape[1])
    idf = np.log(n_samples / df) + 1
    idf_diag = diags(idf, offsets=0, shape=(n_features, n_features), format='csr')
    
    return np.ravel(idf_diag.sum(axis=0))

class corpus:
    """Corpus class
    
    This class regroup the basic NLP tools for corpus processing.

    Attributes:
        data (:obj:`list` of :obj:`str`): The corpus to process.
        stop_words bool: Either to keep stop words or not.
        
    Todo:
        Adapt code for different processing (ner, parser, tagger)

    """
    def __init__(self, data, stop_words=True, disable=['parser', 'tagger', 'ner']):
        self.data = data   
        self.stop_words = stop_words
        self.nlp = spacy.load('en_core_web_sm', disable=disable)
    
    @property
    def tokens(self):
        """:obj:`list` of :obj:`list`: the tokenized corpus. """
        if not hasattr(self, '_tokens'):
            self._tokens = []
            for i, doc in enumerate(tqdm(self.nlp.pipe(self.data, n_threads=-1, batch_size=64), total=len(self.data))):
                self._tokens.append([token.text for token in doc if token.is_stop == self.stop_words])
        return self._tokens
    
    @property
    def vocab(self):
        """:obj:`list` of :obj:`tupple`: the vocabulary of the corpus with the words counts."""
        if not hasattr(self, '_vocab'):
            self._vocab = get_vocab(self.tokens)
        return self._vocab
    
    @property
    def indexed(self):
        """:obj:`list` of :obj:`list`: the indexed corpus."""
        if not hasattr(self, '_indexed'):
            self._indexed = index_corpus(self.tokens, self.vocab)
        return self._indexed
    
    @property
    def BoW(self):
        """sparse matrix: the Bags of Words."""
        if not hasattr(self, '_BoW'):
            self._BoW = to_BoW(self.indexed, self.vocab)
        return self._BoW
    
    @property
    def idf(self):
        """ndarray: the inverse document frequency."""
        if not hasattr(self, '_idf'):
            self._idf = get_idf(self.BoW)
        return self._idf
    