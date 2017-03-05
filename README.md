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

1) An Input Encoder, that takes the input sequence at a given time step, and encodes it into a fixed-size vector representation <img src="https://rawgit.com/siddk/entity-network/None/eval/svgs/1f1c28e0a1b1708c6889fb006c886784.svg?invert_in_darkmode" align=middle width=12.623985pt height=14.10255pt/>

2) The Dynamic Memory (the core of the model), that keeps a disparate set of memory cells, each with a different vector key <img src="https://rawgit.com/siddk/entity-network/None/eval/svgs/40cca55dbe7b8452cf1ede03d21fe3ed.svg?invert_in_darkmode" align=middle width=17.806305pt height=14.10255pt/> (the location), and a hidden state memory <img src="https://rawgit.com/siddk/entity-network/None/eval/svgs/6d22be1359e204374e6f0b45e318d561.svg?invert_in_darkmode" align=middle width=15.517425pt height=22.74591pt/> (the content)

3) The Output Module, that takes the hidden states, and applies a series of transformations to generate the output <img src="https://rawgit.com/siddk/entity-network/None/eval/svgs/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.61696pt height=14.10255pt/>.

A breakdown of the components are as follows:

**Input Encoder**: Takes the input from the environment (i.e. a sentence from a story), and maps it to a fixed size state vector <img src="https://rawgit.com/siddk/entity-network/None/eval/svgs/1f1c28e0a1b1708c6889fb006c886784.svg?invert_in_darkmode" align=middle width=12.623985pt height=14.10255pt/>.

This repository (like the paper) utilizes a learned multiplicative mask, where each embedding of the sentence is multiplied element-wise with a mask vector <img src="https://rawgit.com/siddk/entity-network/None/eval/svgs/9b6dbadab1b122f6d297345e9d3b8dd7.svg?invert_in_darkmode" align=middle width=12.65154pt height=22.74591pt/> and then summed together. 

Alternatively, one could just as easily imagine an LSTM or CNN encoder to generate this initial input.

**Dynamic Memory**: Core of the model, consists of a series of key vectors <img src="https://rawgit.com/siddk/entity-network/None/eval/svgs/5ccebbf530ff52e71bfb606d574fdaca.svg?invert_in_darkmode" align=middle width=98.039205pt height=14.10255pt/> and memory (hidden state) vectors <img src="https://rawgit.com/siddk/entity-network/None/eval/svgs/68db9e670b455c9eef5d6b82287b3676.svg?invert_in_darkmode" align=middle width=91.17273pt height=22.74591pt/>.

The keys and state vectors function similarly to how the program keys and program embeddings function in the NPI/NTM - the keys represent location, while the memories are content.
Only the content (memories) get updated at inference time, with the influx of new information. 

Furthermore, one can seed and fix the key vectors such that they reflect certain words/entities => the paper does this by fixing key vectors to certain word embeddings, and using a simple BoW state encoding.
This repository currently only supports random key vector seeds.

The Dynamic Memory updates given an input <img src="https://rawgit.com/siddk/entity-network/None/eval/svgs/1f1c28e0a1b1708c6889fb006c886784.svg?invert_in_darkmode" align=middle width=12.623985pt height=14.10255pt/> are as follows - this is very similar to the GRU update equations:

+ <img src="https://rawgit.com/siddk/entity-network/None/eval/svgs/e30634013819f430680ff7d9d2d67190.svg?invert_in_darkmode" align=middle width=195.352245pt height=27.59823pt/> 
    - Gating function, determines how much memory j should be affected by the given input.

+ <img src="https://rawgit.com/siddk/entity-network/None/eval/svgs/070eb4a8ad370755d533e0f8c6dea9aa.svg?invert_in_darkmode" align=middle width=259.068645pt height=30.55107pt/> 
    - New state update - U, V, W are model parameters that are shared across all memory cells .
    - Model can be simplified by constraining U, V, W to be zero, or identity.

+ <img src="https://rawgit.com/siddk/entity-network/None/eval/svgs/1f3fc749aea58a01cb3dfe4942924983.svg?invert_in_darkmode" align=middle width=116.74773pt height=30.55107pt/>
    - Gated update, elementwise product of g with $\tilde{h}$.
    - Dictates how much the given memory should be updated.

**Output Module**: Model interface, takes in the memories and a query vector q, and transforms them into the required output.

Functions like a 1-hop Memory Network (Sukhbaatar, Weston), building a weighting mechanism over each input, then combines and feeds them through some intermediate layers. 

The actual updates are as follows:

+ <img src="https://rawgit.com/siddk/entity-network/None/eval/svgs/95f55f07bc9f2f920afdda389426a490.svg?invert_in_darkmode" align=middle width=141.104535pt height=27.59823pt/>
    - Normalizes states based on cosine similarity.
+ <img src="https://rawgit.com/siddk/entity-network/None/eval/svgs/341317311489f34a57af823138e4fd8a.svg?invert_in_darkmode" align=middle width=88.94622pt height=24.65793pt/>
    - Weighted sum of hidden states
+ <img src="https://rawgit.com/siddk/entity-network/None/eval/svgs/0ac5cd1ce89de7dbc1e4329a272457d6.svg?invert_in_darkmode" align=middle width=192.040695pt height=24.56553pt/> 
    - R, H are trainable model parameters.
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
