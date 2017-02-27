# Recurrent Entity Networks
Tensorflow/TFLearn Implementation of ["Tracking the World State with Recurrent Entity Networks"](https://arxiv.org/abs/1612.03969) by Henaff et. al.

### Punchline ###
By building a set of disparate memory cells, each responsible for different concepts, entities, or other content, Recurrent Entity Networks (EntNets) are able to efficiently and robustly maintain a “world-state” - one that can be updated easily and effectively with the influx of new information. 

Furthermore, one can either let EntNet cell keys vary, or specifically seed them with specific embeddings, thereby forcing the model to track a given set of entities/objects/locations, allowing for the easy interpretation of the underlying decision-making process.

### Results ###
Implementation results are as follows (graphs of training/validation loss will be added later). Some of the tasks 
are fairly computationally intensive, so it might take a while to get benchmark results.

1) **Single-Supporting Fact**

    + Test Accuracy: 98.9\%
    
    + Epochs to Converge: 50


### Components ###

Entity Networks consist of three separate components:

1) An Input Encoder, that takes the input sequence at a given time step, and encodes it into a fixed-size vector representation *s_t*

2) The Dynamic Memory (the core of the model), that keeps a disparate set of memory cells, each with a different vector key *w_j* (the location), and a hidden state memory *h_j* (the content)

3) The Output Module, that takes the hidden states, and applies a series of transformations to generate the output *y*.

A breakdown of the components are as follows:

**Input Encoder**: Takes the input from the environment (i.e. a sentence from a story), and maps it to a fixed size state vector *s_t*.

This repository (like the paper) utilizes a learned multiplicative mask, where each embedding of the sentence is multiplied element-wise with a mask vector *f_i* and then summed together. 

Alternatively, one could just as easily imagine an LSTM or CNN encoder to generate this initial input.

**Dynamic Memory**: Core of the model, consists of a series of key vectors *w_1*, *w_2*, ... *w_m* and memory (hidden state) vectors *h_1*, *h_2*, ... *h_m*.

The keys and state vectors function similarly to how the program keys and program embeddings function in the NPI/NTM - the keys represent location, while the memories are content.
Only the content (memories) get updated at inference time, with the influx of new information. 

Furthermore, one can seed and fix the key vectors such that they reflect certain words/entities => the paper does this by fixing key vectors to certain word embeddings, and using a simple BoW state encoding.
This repository currently only supports random key vector seeds.

The Dynamic Memory updates given an input *s_t* are as follows - this is very similar to the GRU update equations:

+ *g_j*  =  sigmoid(*s_t*^T *h_j* + *s_t*^T *w_j*) 
    - Gating function, determines how much memory j should be affected by the given input.

+ ~*h_j*  = activation(**U** *h_j* + **V** *w_j* + **W** *s_t*) 
    - New state update - U, V, W are model parameters that are shared across all memory cells .
    - Model can be simplified by constraining U, V, W to be zero, or identity.

+ *h_j*   =  *h_j*  + *g_j* * *~h_j* 
    - Gated update, elementwise product of g with ~h.
    - Dictates how much the given memory should be updated.

**Output Module**: Model interface, takes in the memories and a query vector q, and transforms them into the required output.

Functions like a 1-hop Memory Network (Sukhbaatar, Weston), building a weighting mechanism over each input, then combines and feeds them through some intermediate layers. 

The actual updates are as follows:

+ *p_j*  =  softmax(*q*^T *h_j*)
    - Normalizes states based on cosine similarity.
+ *u* = ∑ *p_j* *h_j* 
    - Weighted sum of hidden states
+ *y* = **R** activation(*q* + **H** *u*) 
    - **R**, **H** are trainable model parameters.
    - As long as you can build some sort of loss using y, then the entirety of the model is trainable via Backpropagation-Through-Time (BPTT).

### Repository Structure ###
Directory is structured in the following way:

+ model/ - Model definition code, including the definition of the Dynamic Memory Cell.

+ preprocessor/ - Preprocessing code to load and vectorize the bAbI Tasks.

+ tasks/ - Raw bAbI Task files.

+ run.py - Core script for training and evaluating the Recurrent Entity Network. 

### References ###
Big shout-out to Jim Fleming for his initial Tensorflow Implementation - his Dynamic Memory Cell Implementation 
specifically made things a lot easier.

Reference: [Jim Fleming's EntNet Memory Cell](https://github.com/jimfleming/recurrent-entity-networks/blob/master/entity_networks/dynamic_memory_cell.py)
