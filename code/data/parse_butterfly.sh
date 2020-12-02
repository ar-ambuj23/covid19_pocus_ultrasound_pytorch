#!/bin/bash

unzip *.zip -d butterfly
python ../scripts/util/process_butterfly_videos.py
python ../scripts/util/build_image_dataset.py
