import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Visualization functions.

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(10, 4))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
    plt.show()

def plot_log(array, maxlim, ylabel, xlabel, display, save):
    fig_1 = plt.figure(figsize=(1.5,1.5))
    plt.ylim(0,maxlim)
    plt.ylabel(ylabel, fontsize=11)
    plt.xlabel(xlabel, fontsize=11)
    plt.plot(array)
    if display:
        plt.show()
    if save:
        plt.savefig('Plots/'+'dl_epoch_'+str(i), bbox_inches='tight')


# Data management functions.

def save_df(logs, output, save):
    lenght = len (logs)
    df = pd.DataFrame(data=None)
    df["Log"] = logs

    if save:
        df.to_csv("Data/Csv/" +output+".csv")


