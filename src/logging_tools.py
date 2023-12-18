import numpy as np
import matplotlib.pyplot as plt
from src.NeuralNetwork import CudaLayer


def layer_agl_plot(layer: CudaLayer, y, loss):
    log = layer.state()
    fig, (*axes,  ax0, ax1) = plt.subplots(1, 5, figsize=(22, 5))
    ax0.plot(layer.history['w'], loss, label='loss')
    min_loss = loss[np.argsort(loss)[0]]
    min_weight = layer.history['w'][np.argsort(loss)[0]]
    ax0.axvline(min_weight, color='r', linestyle=':')
    ax0.axhline(min_loss, color='r', linestyle=':')
    ax0.scatter(min_weight, min_loss, marker='^', color='g', zorder=10)
    ax0.scatter(layer.history['w_init'], loss[0], marker='o', color='purple', zorder=5)
    ax0.scatter(layer.history['w'][::len(loss) // 10][1:], loss[::len(loss) // 10][1:], marker='o',
                color='tab:orange', zorder=2)
    ax0.legend()

    ax1.plot(loss)
    image = axes[0].imshow(log[1])
    fig.colorbar(image, ax=axes[0])
    for i in range(log[0].shape[-1]):
        axes[1].plot(sorted(log[0][:, i]), linestyle=':')
        tmp = axes[1].twiny()
        tmp.plot(sorted(log[0].flatten()), zorder=10, linewidth=2, color='green')

    values = [sum(np.abs(y.flatten())), sum(np.abs(log[0].flatten()))]
    top = np.argmax(values)
    axes[2].hist(log[0].flatten(), label='activation', color='red', zorder=[10, 0][top])
    axes[2].hist(y.flatten(), label='label', zorder=[0, 10][top])
    axes[2].legend()
    axes[2].set_title('Activation Histogram')
    axes[0].set_title('Weights')
    axes[1].set_title('Activation Graph')
    ax0.set_title('Gradient')
    ax0.set_xlabel('norm(weights)')
    ax0.set_ylabel('loss')
    ax1.set_title('Loss')
    ax1.set_ylabel('loss')
    axes[2].set_ylabel('count')
    axes[2].set_xlabel('value')
    axes[0].set_ylabel('Input')
    axes[0].set_xlabel('Output')
    axes[1].set_ylabel('Value')
    plt.tight_layout()
    plt.show()