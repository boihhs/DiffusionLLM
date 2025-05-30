import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from dataload import CharTokenizer, chars_by_freq
from matplotlib.animation import PillowWriter

def animate_diffusion_states(states_ids, tokenizer, interval=300,
                             tokens_per_line=None,
                             cell_w=0.3,   # width of one cell in data units
                             cell_h=0.4):  # height of one cell in data units
    states = np.array(states_ids)
    T, B = states.shape

    if tokens_per_line is None:
        tokens_per_line = int(np.ceil(np.sqrt(B)))
    lines = int(np.ceil(B / tokens_per_line))

    # decode as before…
    decoded = []
    for t in range(T):
        row = []
        for b in range(B):
            tok = tokenizer.decode([int(states[t, b])]).strip()
            row.append(tok if len(tok)==1 else ' ')
        decoded.append(row)

    # set up figure so that each data-unit is small on screen
    fig_w = tokens_per_line * cell_w   # inches
    fig_h = lines            * cell_h  # inches
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    # set the data limits so that 0→ tokens_per_line-1 covers tokens_per_line cells
    ax.set_xlim(-cell_w/2, tokens_per_line*cell_w - cell_w/2)
    ax.set_ylim(-cell_h/2, lines*cell_h - cell_h/2)

    texts = []
    for idx in range(B):
        row = idx // tokens_per_line
        col = idx %  tokens_per_line
        # invert row so that row=0 is top
        y = (lines - 1 - row) * cell_h
        x = col * cell_w
        txt = ax.text(x, y, '', fontsize=8, fontfamily='monospace',
                      ha='center', va='center')
        texts.append(txt)

    def update(frame):
        for idx, txt in enumerate(texts):
            txt.set_text(decoded[frame][idx])
        return texts

    anim = FuncAnimation(fig, update, frames=T,
                         interval=interval, blit=True, repeat=False)
    out_path = "diffusion.gif"
    fps = 1000 / interval
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.show()
    return anim



# tokenizer = CharTokenizer(chars_by_freq)
# states_ids = [
#     [0,1,2,3,4,0,0,0,0,0],
#     [1,1,2,3,4,0,0,0,0,0],
#     [1,2,2,3,4,0,0,0,0,0],
#     [1,2,3,3,4,0,0,0,0,0],
#     [1,2,3,4,4,0,0,0,0,0],
#     [1,2,3,4,5,6,7,8,0,0],
# ]

# # Play animation
# animate_diffusion_states(states_ids, tokenizer, interval=200, tokens_per_line=5, cell_w=0.1, cell_h= .2)
