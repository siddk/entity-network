"""
reader.py

Core script containing preprocessing logic - reads bAbI Task Story, and returns
vectorized forms of the stories, questions, and answers.
"""
import numpy as np
import os
import pickle
import re

FORMAT_STR = "qa%d_"
PAD_ID = 0
SPLIT_RE = re.compile('(\W+)?')

def parse(data_path, task_id, data_type, word2id=None, bsz=32):
    cache_path = data_path + "-pik/" + FORMAT_STR % task_id + data_type + ".pik"
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return pickle.load(f)
    else:
        [filename] = filter(lambda x: FORMAT_STR % task_id in x and data_type in x, os.listdir(data_path))
        S, S_len, Q, A, word2id = parse_stories(os.path.join(data_path, filename), word2id)
        n = (S.shape[0] / bsz) * bsz
        with open(cache_path, 'w') as f:
            pickle.dump((S[:n], S_len[:n], Q[:n], A[:n], word2id), f)
        return S[:n], S_len[:n], Q[:n], A[:n], word2id

def parse_stories(filename, word2id=None):
    # Open file, get lines
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Go through lines, building story sets
    stories, story = [], []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            query, answer, supporting = line.split('\t')
            query = tokenize(query)
            substory = [x for x in story if x]
            stories.append((substory, query, answer))
            story.append('')
        else:
            sentence = tokenize(line)
            story.append(sentence)
    
    # Build Vocabulary
    if not word2id:
        vocab = set(reduce(lambda x, y: x + y, [q for (_, q, _) in stories]))
        for (s, _, _) in stories:
            for sentence in s:
                vocab.update(sentence)
        id2word = ['PAD_ID'] + list(vocab)
        word2id = {w: i for i, w in enumerate(id2word)}
    
    # Get Maximum Lengths
    sentence_max, story_max = 0, 0
    for (s, q, _) in stories: 
        if len(q) > sentence_max:
            sentence_max = len(q)
        if len(s) > story_max:
            story_max = len(s)
        for sentence in s:
            if len(sentence) > sentence_max:
                sentence_max = len(sentence)

    # Allocate Arrays
    S = np.zeros([len(stories), story_max, sentence_max], dtype=np.int32)
    Q = np.zeros([len(stories), sentence_max], dtype=np.int32)
    S_len, A = np.zeros([len(stories)], dtype=np.int32), np.zeros([len(stories)], dtype=np.int32)

    # Fill Arrays
    for i, (s, q, a) in enumerate(stories):
        # Populate story
        for j in range(len(s)):
            for k in range(len(s[j])):
                S[i][j][k] = word2id[s[j][k]]

        # Populate story length
        S_len[i] = len(s)
        
        # Populate Question
        for j in range(len(q)):
            Q[i][j] = word2id[q[j]]
        
        # Populate Answer 
        A[i] = word2id[a]
        
    return S, S_len, Q, A, word2id

def tokenize(sentence):
    """
    Tokenize a string by splitting on non-word characters and stripping whitespace.
    """
    return [token.strip().lower() for token in re.split(SPLIT_RE, sentence) if token.strip()]