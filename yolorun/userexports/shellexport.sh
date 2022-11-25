#!/usr/bin/env bash
declare -a models=("n" "s" "m" "l" "x" "n6" "s6" "m6" "l6" "x6")
size=1280
batch=3

echo "Making the folder for models"
mkdir models_"$size"x"$size"_batch_"$batch"
echo "Starting model exports"
for i in "${models[@]}"
do
   python ../export.py --weights yolov5"$i".pt --batch-size "$batch" --include onnx --imgsz "$size"
done
echo "Moving exported models to user specified folder"
mv yolov5*.* models_"$size"x"$size"_batch_"$batch"/
echo "Done!"