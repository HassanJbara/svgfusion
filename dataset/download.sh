#!/usr/bin/env bash

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

echo "Downloading dataset..."
gdrive_download 10Zx4TB1-BEdWv1GbwcSUl2-uRFiqgUP1 ./dataset/icons_meta.csv
gdrive_download 1gTuO3k98u_Y1rvpSbJFbqgCf6AJi2qIA ./dataset/icons_tensor.zip

echo "Download done. Unzipping..."
unzip ./dataset/icons_tensor.zip -d ./dataset/
echo "Done."
