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
DATA_TYPES = ['train', 'valid', 'test']

def parse(data_path, task_id, word2id=None, bsz=32):
    vectorized_data, story_data, global_sentence_max, global_story_max = [], [], 0, 0
    for data_type in DATA_TYPES:
        cache_path = data_path + "-pik/" + FORMAT_STR % task_id + data_type + ".pik"
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                vectorized_data.append(pickle.load(f))
        else:
            [filename] = filter(lambda x: FORMAT_STR % task_id in x and data_type in x, os.listdir(data_path))
            stories, sentence_max, story_max, word2id = parse_stories(os.path.join(data_path, filename), word2id)
            story_data.append(stories)
            global_sentence_max = max(global_sentence_max, sentence_max)
            global_story_max = max(global_story_max, story_max)
    if vectorized_data:
        return vectorized_data + [vectorized_data[0][4]]
    else:
        for i, data_type in enumerate(DATA_TYPES):
            cache_path = data_path + "-pik/" + FORMAT_STR % task_id + data_type + ".pik"
            S, S_len, Q, A = vectorize_stories(story_data[i], global_sentence_max, global_story_max, word2id, task_id)
            n = (S.shape[0] / bsz) * bsz
            with open(cache_path, 'w') as f:
                pickle.dump((S[:n], S_len[:n], Q[:n], A[:n], word2id), f)
            vectorized_data.append((S[:n], S_len[:n], Q[:n], A[:n], word2id))
        return vectorized_data + [word2id]

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
    
    return stories, sentence_max, story_max, word2id

def vectorize_stories(stories, sentence_max, story_max, word2id, task_id):
    # Check Story Max 
    if task_id == 3:
        story_max = min(story_max, 130)
    else:
        story_max = min(story_max, 70)

    # Allocate Arrays
    S = np.zeros([len(stories), story_max, sentence_max], dtype=np.int32)
    Q = np.zeros([len(stories), sentence_max], dtype=np.int32)
    S_len, A = np.zeros([len(stories)], dtype=np.int32), np.zeros([len(stories)], dtype=np.int32)

    # Fill Arrays
    for i, (s, q, a) in enumerate(stories):
        # Check S Length => All but Task 3 are limited to 70 sentences
        if task_id == 3:
             s = s[-130:]
        else:
            s = s[-70:]
        
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
        
    return S, S_len, Q, A

def tokenize(sentence):
    """
    Tokenize a string by splitting on non-word characters and stripping whitespace.
    """
    return [token.strip().lower() for token in re.split(SPLIT_RE, sentence) if token.strip()]