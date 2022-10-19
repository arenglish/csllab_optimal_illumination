from matplotlib import rcParams, animation, pyplot as plt
from numpy import linspace, sin, array, c_, amin
from IPython import display

rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'


def animate(fig, fn, iterations, name=None, fps=30):
    anim = animation.FuncAnimation(fig, fn,
                                   frames=iterations, interval=100, repeat=True, save_count=iterations)
    if name is not None:
        writervideo = animation.FFMpegWriter(fps=fps)
        anim.save(name, writer=writervideo)

    plt.close()
