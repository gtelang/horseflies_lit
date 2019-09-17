ffmpeg -r 10 -i plot_%05d.png -s 1380x720 -c:v libx264 -qscale 10 -r 30 distributed_rhf.mp4
