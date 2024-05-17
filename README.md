# deepfake-detection-with-xception


训练：执行下面的命令，dataset文件夹下是fake和real文件夹，其中放入相应的图片用于训练
```bash
python train_dataset.py dataset/ classes.txt  result/
```

Predefined settings:
```bash
Epoch: 10 / 30 [First/Final stage]
Learning rate: 5e-3 / 5e-4
Batch size: 32 / 64
```

- Then take the best model from examining the graph and run `app.py` to detect videos. It can take a video file or a youtube-dl supported video link as a input. Note that we've tested online links only with Youtube so your results may vary.

调用模型进行检测，models/x-model23.p 是保存的model的路径， video_path是需要检测的视频文件路径, 另外还需要创建一个image文件夹，用来放视频文件提取的图片帧
```bash
python image_prediction.py models/x-model23.p classes.txt video_path
```

