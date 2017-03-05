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

1) An Input Encoder, that takes the input sequence at a given time step, and encodes it into a fixed-size vector representation <p align="center"><img src="https://rawgit.com/siddk/entity-network/master/eval/svgs/9da86ae1c0aa2293db8f7d07ab186bf5.svg?invert_in_darkmode" align=middle width=12.6239355pt height=9.516903pt/></p>

2) The Dynamic Memory (the core of the model), that keeps a disparate set of memory cells, each with a different vector key <p align="center"><img src="https://rawgit.com/siddk/entity-network/master/eval/svgs/98d5abbafdc10c296ee92084f5c83ed4.svg?invert_in_darkmode" align=middle width=17.806305pt height=11.745987pt/></p> (the location), and a hidden state memory <p align="center"><img src="https://rawgit.com/siddk/entity-network/master/eval/svgs/bc0dfb3ea407c81c1ce46b2d59ed2be7.svg?invert_in_darkmode" align=middle width=15.5174415pt height=16.0677pt/></p> (the content)

3) The Output Module, that takes the hidden states, and applies a series of transformations to generate the output *y*.

A breakdown of the components are as follows:

**Input Encoder**: Takes the input from the environment (i.e. a sentence from a story), and maps it to a fixed size state vector <p align="center"><img src="https://rawgit.com/siddk/entity-network/master/eval/svgs/9da86ae1c0aa2293db8f7d07ab186bf5.svg?invert_in_darkmode" align=middle width=12.6239355pt height=9.516903pt/></p>.

This repository (like the paper) utilizes a learned multiplicative mask, where each embedding of the sentence is multiplied element-wise with a mask vector <p align="center"><img src="https://rawgit.com/siddk/entity-network/master/eval/svgs/1fa89784462d8bb8d403d245c34983c0.svg?invert_in_darkmode" align=middle width=12.651441pt height=14.55729pt/></p> and then summed together. 

Alternatively, one could just as easily imagine an LSTM or CNN encoder to generate this initial input.

**Dynamic Memory**: Core of the model, consists of a series of key vectors <p align="center"><img src="https://rawgit.com/siddk/entity-network/master/eval/svgs/311a7f69b3c159624af52e06096b214a.svg?invert_in_darkmode" align=middle width=98.03904pt height=10.2355935pt/></p> and memory (hidden state) vectors <p align="center"><img src="https://rawgit.com/siddk/entity-network/master/eval/svgs/7bd8f7862e970280c1768c7cbd951b79.svg?invert_in_darkmode" align=middle width=91.17273pt height=14.55729pt/></p>.

The keys and state vectors function similarly to how the program keys and program embeddings function in the NPI/NTM - the keys represent location, while the memories are content.
Only the content (memories) get updated at inference time, with the influx of new information. 

Furthermore, one can seed and fix the key vectors such that they reflect certain words/entities => the paper does this by fixing key vectors to certain word embeddings, and using a simple BoW state encoding.
This repository currently only supports random key vector seeds.

The Dynamic Memory updates given an input <p align="center"><img src="https://rawgit.com/siddk/entity-network/master/eval/svgs/9da86ae1c0aa2293db8f7d07ab186bf5.svg?invert_in_darkmode" align=middle width=12.6239355pt height=9.516903pt/></p> are as follows - this is very similar to the GRU update equations:

+ <p align="center"><img src="https://rawgit.com/siddk/entity-network/master/eval/svgs/d8f3196c7a4116939acc0dbcccc56a47.svg?invert_in_darkmode" align=middle width=195.35175pt height=19.315725pt/></p> 
    - Gating function, determines how much memory j should be affected by the given input.

+ <p align="center"><img src="https://rawgit.com/siddk/entity-network/master/eval/svgs/38a2d4299891ce54233e21ae056bb7dd.svg?invert_in_darkmode" align=middle width=259.06815pt height=19.97028pt/></p> 
    - New state update - U, V, W are model parameters that are shared across all memory cells .
    - Model can be simplified by constraining U, V, W to be zero, or identity.

+ <p align="center"><img src="https://rawgit.com/siddk/entity-network/master/eval/svgs/4157c964a1356cfa98d96ac886e630b8.svg?invert_in_darkmode" align=middle width=116.747565pt height=19.97028pt/></p>
    - Gated update, elementwise product of g with ~h.
    - Dictates how much the given memory should be updated.

**Output Module**: Model interface, takes in the memories and a query vector q, and transforms them into the required output.

Functions like a 1-hop Memory Network (Sukhbaatar, Weston), building a weighting mechanism over each input, then combines and feeds them through some intermediate layers. 

The actual updates are as follows:

+ <p align="center"><img src="https://rawgit.com/siddk/entity-network/master/eval/svgs/7a3f0df1ef826e090dbae54abe948537.svg?invert_in_darkmode" align=middle width=141.104535pt height=19.315725pt/></p>
    - Normalizes states based on cosine similarity.
+ <p align="center"><img src="https://rawgit.com/siddk/entity-network/master/eval/svgs/464c556755cc1642c645a44e763ac409.svg?invert_in_darkmode" align=middle width=88.412445pt height=38.878455pt/></p>
    - Weighted sum of hidden states
+ <p align="center"><img src="https://rawgit.com/siddk/entity-network/master/eval/svgs/9e8ce0d2c9b7d64d9cff2cc2f3621f11.svg?invert_in_darkmode" align=middle width=180.1866pt height=16.376943pt/></p> 
    - $$\mathbf{R}, \mathbf{H}$$ are trainable model parameters.
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
