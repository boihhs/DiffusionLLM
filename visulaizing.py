import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def animate_diffusion_states(states_ids, tokenizer, interval=300, tokens_per_line=None,
                             cell_w=0.3, cell_h=0.4):
    states = np.array(states_ids)
    T, B = states.shape
    if tokens_per_line is None:
        tokens_per_line = int(np.ceil(np.sqrt(B)))
    lines = int(np.ceil(B / tokens_per_line))

    decoded = []
    for t in range(T):
        row = []
        for b in range(B):
            tok = tokenizer.decode([int(states[t, b])]).strip()
            row.append(tok if len(tok) == 1 else ' ')
        decoded.append(row)

    fig_w = tokens_per_line * cell_w
    fig_h = lines * cell_h
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    ax.set_xlim(-cell_w/2, tokens_per_line*cell_w - cell_w/2)
    ax.set_ylim(-cell_h/2, lines*cell_h - cell_h/2)

    texts = []
    for idx in range(B):
        row = idx // tokens_per_line
        col = idx % tokens_per_line
        y = (lines - 1 - row) * cell_h
        x = col * cell_w
        txt = ax.text(x, y, '', fontsize=8, fontfamily='monospace',
                      ha='center', va='center')
        texts.append(txt)

    def update(frame):
        for idx, txt in enumerate(texts):
            txt.set_text(decoded[frame][idx])
        return texts

    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=True, repeat=False)
    out_path = "diffusion.gif"
    fps = 1000 / interval
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.show()
    return anim
