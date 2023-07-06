import matplotlib.pyplot as plt

def plot_curve(y, title, x_label, y_label):
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return fig