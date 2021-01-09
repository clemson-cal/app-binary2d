#!/usr/bin/env python3

import os

os.system("ffmpeg -f image2 -r 24 -i /Users/jackhu/Google_Drive/Rust_personal"
          "/projects/app-binary2d/movies/chkpt.%04d.h5.jpg -vcodec mpeg4 "
          "-y /Users/jackhu/Google_Drive/Rust_personal/projects/app-binary2d"
          "/movies/movie.mp4")

