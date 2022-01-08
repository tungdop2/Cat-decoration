# Real-time cat decoration
This model is based on landmark detection with ResNet18 backbone.
Currently just work for 1 cat per image, and still wear glasses, because model just return position, and no confidence.
# Run
1. Install the required packages
   ```
   pip install requirements.txt
   ```
2. Run real-time decorating with command
   ```
   python decorate.py 
                    --source <0 for webcam, string for video path>
                    --save_path <path to save the output video, default is ./output.mp4>
                    --show_size <output video size, default is 640x480>
   ```

# Demo
![demo](demo/output.gif)

# Coming soon
1. Detecting multiple cats
2. Improve the performance and accuracy
3. Mode decorate options