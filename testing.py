<<<<<<< HEAD
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np

def animate_decoding(tokens, interval=300):
    """
    Animate the decoding of tokens: each token occupies the same space,
    special tokens (length > 1) are displayed as spaces, and letters
    are revealed sequentially.
    
    tokens: list of strings (each string is a token)
    interval: delay between frames in milliseconds
    """
    # Prepare display characters: single-char tokens keep their char, others become space
    final_chars = [tok if len(tok) == 1 else ' ' for tok in tokens]
    N = len(final_chars)
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(N * 0.3, 1.5))
    ax.axis('off')
    text = ax.text(0.5, 0.5, ' ' * N, fontsize=24, family='monospace',
                   ha='center', va='center')
    
    def update(frame):
        # Reveal characters up to the current frame
        display_chars = [' ']*N
        for i in range(frame + 1):
            display_chars[i] = final_chars[i]
        text.set_text(''.join(display_chars))
        return text,
    
    anim = FuncAnimation(fig, update, frames=N, interval=interval, blit=True, repeat=False)
    plt.show()

    plt.close(fig)
    return HTML(anim.to_jshtml())

# Example usage: replace or extend this list with your decoding tokens
tokens = ['<s>', 'H', 'e', 'l', 'l', 'o', '<mask>', 'w', 'o', 'r', 'l', 'd', '!', '</s>']
animation = animate_decoding(tokens, interval=400)


=======
import matplotlib.pyplot as plt
import torch

# Manually extracted tensor values (truncated for brevity here)
loss_values = [
    927820.25, 761230.4375, 748531.5625, 734250.875, 761949.5, 751894.125, 743633.875,
    745200.625, 744908.875, 755074.875, 744152.25, 763122.625, 736337.5625, 770134.75,
    752632.8125, 751939.25, 798521.0, 750251.5, 758114.125, 745662.125, 753453.0625,
    739388.625, 745903.9375, 757280.875, 745869.1875, 755257.5, 746650.9375, 757955.9375,
    736810.4375, 743014.5, 746159.5625, 734347.5625, 745023.25, 743393.5, 736305.5625,
    745663.0, 741839.75, 754595.75, 746143.25, 748292.25, 745355.0625, 746081.6875,
    748637.25, 754170.625, 741203.75, 756557.5, 744205.4375, 735753.75, 753629.6875,
    752784.5, 759461.375, 766749.125, 746391.5625, 736539.75, 748401.125, 744211.5625,
    753527.6875, 756592.625, 740495.1875, 752841.125, 769369.3125, 748145.875, 764122.1875,
    744883.5, 757200.75, 756320.6875, 751817.375, 753173.75, 739179.8125, 758329.25,
    758771.0625, 754758.0, 742595.4375, 748671.375, 761191.25, 746896.5, 752098.0625,
    757150.25, 755704.75, 735232.3125, 749260.4375, 754464.75, 728144.5, 753522.9375,
    737913.75, 743710.0, 753384.875, 740474.25, 761093.125, 754338.5625, 741767.75,
    748741.125, 738405.875, 747485.125, 747631.625, 784733.375, 741484.625, 749108.75,
    759112.9375, 739986.8125, 735354.0625, 743196.25, 741711.875, 740253.625, 748521.375,
    740989.375, 746669.75, 730440.0, 754843.75, 750553.875, 751678.75, 750421.6875,
    753507.0, 758237.4375, 761675.0, 758847.25, 751594.4375, 747376.375, 738957.875,
    749404.8125, 765350.375, 740724.0, 747222.625, 740493.75, 756010.0, 752088.9375,
    749340.75, 759054.375, 734456.9375, 760938.125, 744018.75, 746225.0, 755044.6875,
    740299.5, 749834.75, 745634.0, 751453.4375, 746618.5625, 751269.5625, 741747.4375,
    739870.375, 745832.1875, 747437.4375, 736917.6875, 754887.875, 735420.9375, 746895.5,
    754642.4375, 758255.5625, 758135.0, 746958.6875, 756071.5, 752394.375, 756625.3125,
    768402.5625, 756341.0625, 762011.0, 747456.0, 738429.1875, 739811.0625, 755153.5,
    757397.5, 755611.125, 765400.0625, 736944.75, 746226.125, 747394.8125, 760217.4375,
    752707.375, 754343.125, 745416.0, 743583.75, 750023.25, 742094.125, 746575.125,
    751213.75, 757986.875, 752068.375, 737199.0, 752863.5, 742630.4375, 740969.4375,
    772028.125, 743274.9375, 740780.625, 742872.375, 737430.6875, 745398.9375, 747285.25,
    752586.5, 756942.5625, 748700.8125, 757025.1875
]

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(loss_values, marker='o', linestyle='-', markersize=2)
plt.title("Loss Over Time")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
>>>>>>> 5ef3fe9be88c49d2ae7bd10b46754207eb21c73c
