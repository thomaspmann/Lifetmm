#######################      NOTE      #######################
# Easier to just use FFMPEG:
# ffmpeg -framerate 20 -i %5d.png -s:v 1280x720 -c:v libx264 \
# -profile:v high -crf 20 -pix_fmt yuv420p vid.mp4
##############################################################

import glob
import os

import matplotlib.animation as animation
import matplotlib.image as mgimg
import matplotlib.pyplot as plt

fig = plt.figure()

ims = []
for i in sorted(glob.glob("../Images/Vary thickness/*.png"), key=os.path.getmtime):
    img = mgimg.imread(i)
    im = plt.imshow(img, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True,
                                repeat_delay=False)

from matplotlib.animation import FFMpegWriter

plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\mn14tm\\Code\\Lifetmm\\Data\\ffmpeg.exe'
writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

ani.save("movie.mp4", writer=writer)

plt.show()

