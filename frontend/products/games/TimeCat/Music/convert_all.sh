#!/usr/bin/env bash

# Target MP3 bitrate (adjust if you want different quality)
BITRATE="128k"

for f in *.wav; do
  # If no .wav files, skip
  [ -e "$f" ] || continue

  name="${f%.*}.mp3"

  echo "Converting: $f → $name"
  start ffmpeg -i "$f" -codec:a libmp3lame -b:a "$BITRATE" "$name"

  if [ -f "$name" ]; then
    echo "Deleting original: $f"
    rm "$f"
  else
    echo "Conversion failed for $f – keeping original"
  fi
done

echo "Done!"
