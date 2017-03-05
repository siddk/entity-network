"""
plot.py

Generates plots of training loss/validation loss over training epochs. Useful
for analyzing performance of the Entity Network.
"""
import pickle
import sys

def plot(task_id):
    with open('../checkpoints/qa_%d/training_logs.pik' % task_id, 'r') as f:
        train_loss, train_acc, val_loss, val_acc = pickle.load(f)
    
    

if __name__ == "__main__":
    task = int(sys.argv[1])
    plot(task)