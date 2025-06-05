import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def animate_decoding(tokens, interval=300):
    """Animate gradual reveal of decoded tokens."""
    final_chars = [tok if len(tok) == 1 else ' ' for tok in tokens]
    N = len(final_chars)
    fig, ax = plt.subplots(figsize=(N * 0.3, 1.5))
    ax.axis('off')
    text = ax.text(0.5, 0.5, ' ' * N, fontsize=24, family='monospace',
                   ha='center', va='center')

    def update(frame):
        display_chars = [' '] * N
        for i in range(frame + 1):
            display_chars[i] = final_chars[i]
        text.set_text(''.join(display_chars))
        return text,

    anim = FuncAnimation(fig, update, frames=N, interval=interval, blit=True, repeat=False)
    plt.show()
    plt.close(fig)
    return HTML(anim.to_jshtml())
