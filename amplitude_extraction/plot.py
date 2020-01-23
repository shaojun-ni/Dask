import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_slice(data, title=''):
    gs = plt.GridSpec(1, 1)
    fig = plt.figure(figsize=(10, 20))
    ax1 = fig.add_subplot(gs[-1])

    # Plot the section
    im = ax1.imshow(data.T, cmap='gist_rainbow', origin='upper')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    ax1.set(title=title)
    plt.show()
    
def plot_horizon(data, title=''):  
    gs = plt.GridSpec(1, 1)
    fig = plt.figure(figsize=(8, 9))
    ax1 = fig.add_subplot(gs[-1])
    
    # Plot the section
    im = ax1.imshow(data.T, cmap='gist_rainbow', origin='lower')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im,cax=cax)
    ax1.set(title=title)
    plt.show()




