"""
plot.py

Generates plots of training loss/validation loss over training epochs. Useful
for analyzing performance of the Entity Network.
"""
import matplotlib.pyplot as plt
import pickle
import sys

def plot(task_id, run_id):
    with open('../checkpoints/qa_%d/training_logs.pik' % task_id, 'r') as f:
        train_loss, train_acc, val_loss, val_acc = pickle.load(f)
    tr_x_loss, tr_y_loss, val_x_loss, val_y_loss = [], [], [], []
    
    # Collect
    for t in sorted(train_loss):
        tr_x_loss.append(t)
        tr_y_loss.append(train_loss[t])
    for v in sorted(val_loss):
        val_x_loss.append(v)
        val_y_loss.append(val_loss[v])
    tr_x_acc, tr_y_acc, val_x_acc, val_y_acc = [], [], [], []
    for t in sorted(train_acc):
        tr_x_acc.append(t)
        tr_y_acc.append(train_acc[t])
    for v in sorted(val_acc):
        val_x_acc.append(v)
        val_y_acc.append(val_acc[v])
    
    # Plot 
    plt.figure(1, figsize=(14, 7))
    plt.subplot(121)
    plt.plot(tr_x_loss, tr_y_loss, 'b-', val_x_loss, val_y_loss, 'g-.')
    plt.title("[Task %d] Training and Validation Loss" % (task_id))
    plt.xlabel('Training Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend(['Training', 'Validation'])
    
    plt.subplot(122)
    plt.plot(tr_x_acc, tr_y_acc, 'b-', val_x_acc, val_y_acc, 'g-.')
    plt.title("[Task %d] Training and Validation Accuracy" % (task_id))
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])
    plt.tight_layout()
    plt.savefig('qa_%d/run_%d.png' % (task_id, run_id))

if __name__ == "__main__":
    task, run = int(sys.argv[1]), int(sys.argv[2])
    plot(task, run)