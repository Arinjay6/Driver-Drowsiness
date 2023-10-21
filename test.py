import time
from pygame import mixer
mixer.init()
mixer.music.load(r"C:\Users\DELL\Downloads\project6sem\project6sem\sound_files\alarm.mp3")
mixer.music.play()
while mixer.music.get_busy():  # wait for music to finish playing
    time.sleep(1)