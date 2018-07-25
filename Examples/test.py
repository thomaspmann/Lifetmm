import glob
import os

for int, i in enumerate(sorted(glob.glob("../Images/Vary thickness/*.png"), key=os.path.getmtime)):
    os.rename(i, '../Images/New/{:}.png'.format(int))
