# import argparse
# import numpy as np
# from keras.applications.xception import preprocess_input
# from tensorflow.keras.preprocessing import image
# from keras.models import load_model

# parser = argparse.ArgumentParser()
# parser.add_argument('model')
# parser.add_argument('classes')
# parser.add_argument('image')
# parser.add_argument('--top_n', type=int, default=2)


# def main(args):

#     model = load_model(args.model)

#     classes = []
#     with open(args.classes, 'r') as f:
#         classes = list(map(lambda x: x.strip(), f.readlines()))

#     img = image.load_img(args.image, target_size=(299, 299))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)

#     # predict
#     pred = model.predict(x)[0]
#     result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
#     result.sort(reverse=True, key=lambda x: x[1])
#     for i in range(args.top_n):
#         (class_name, prob) = result[i]
#         print("Top %d =" % (i + 1))
#         print("Class: %s" % (class_name))
#         print("Probability: %.2f%%" % (prob))


# if __name__ == '__main__':
#     args = parser.parse_args()
#     main(args)



# import argparse
# import numpy as np
# from keras.applications.xception import preprocess_input
# from tensorflow.keras.preprocessing import image
# from keras.models import load_model
# import os

# parser = argparse.ArgumentParser()
# parser.add_argument('model')
# parser.add_argument('classes')
# # parser.add_argument('videopath')
# parser.add_argument('--imagepath',default='image/')
# parser.add_argument('--top_n', type=int, default=2)

# real_num = 0
# fake_num = 0

# def main(args, ima):
#     global real_num, fake_num
#     model = load_model(args.model)

#     classes = []
#     with open(args.classes, 'r') as f:
#         classes = list(map(lambda x: x.strip(), f.readlines()))

#     img = image.load_img(f"{args.imagepath}{ima}", target_size=(299, 299))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)

#     # predict
#     pred = model.predict(x)[0]
#     result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
#     result.sort(reverse=True, key=lambda x: x[1])
#     class_name, _ = result[0]
#     if class_name == "real":
#         real_num = real_num + 1
#     else:
#         fake_num = fake_num + 1

#     for i in range(args.top_n):
#         (class_name, prob) = result[i]
#         print("Top %d =" % (i + 1))
#         print("Class: %s" % (class_name))
#         print("Probability: %.2f%%" % (prob))


# if __name__ == '__main__':
#     args = parser.parse_args()
#     path_list = os.listdir(f"{args.imagepath}")
#     for path in path_list:
#         main(args, path)
#     if real_num > fake_num:
#         print("Real video")
#     else:
#         print("Fake video")
#     # main(args)


import argparse
import numpy as np
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import cv2
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('classes')
parser.add_argument('videopath')
parser.add_argument('--imagepath',default='image')
parser.add_argument('--top_n', type=int, default=2)

real_num = 0
fake_num = 0

def main(args, ima):
    global real_num, fake_num
    model = load_model(args.model)

    classes = ['fake', 'real']
    # with open(args.classes, 'r') as f:
    #     classes = list(map(lambda x: x.strip(), f.readlines()))
    # print(classes)

    img = image.load_img(f"{args.imagepath}/{ima}", target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)


    # predict
    pred = model.predict(x)[0]
    result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
    result.sort(reverse=True, key=lambda x: x[1])
    class_name, _ = result[0]
    if class_name == "real":
        real_num = real_num + 1
    else:
        fake_num = fake_num + 1

    for i in range(args.top_n):
        (class_name, prob) = result[i]
        print("Top %d =" % (i + 1))
        print("Class: %s" % (class_name))
        print("Probability: %.2f%%" % (prob))


divide_photo_num = 0

def save_image(image, addr):
    global divide_photo_num
    address = addr + str(divide_photo_num) + '.jpg'
    divide_photo_num = divide_photo_num + 1
    cv2.imwrite(address, image)

def get_photo(path_mp4, path_image):
    global num
    # 读取视频文件
    videoCapture = cv2.VideoCapture(f"{path_mp4}")
    # 通过摄像头的方式
    # videoCapture=cv2.VideoCapture(1)

    # 读帧
    success, frame = videoCapture.read()
    i = 0
    timeF = 120
    j = 0
    while success:
        i = i + 1
        if (i % timeF == 0):
            j = j + 1
            save_image(frame, f'{path_image}/image')
            print('save image:', i)
        success, frame = videoCapture.read()


if __name__ == '__main__':
    args = parser.parse_args()
    get_photo(f"{args.videopath}", f"{args.imagepath}")
    path_list = os.listdir(f"{args.imagepath}")
    # path_list.remove(".ipynb_checkpoints")
    for path in path_list:
        main(args, path)
    for path in path_list:
        os.remove(f"{args.imagepath}/{path}")
    
    print("*" * 50)
    if real_num > fake_num:
        print("Real video")
    else:
        print("Fake video")
    # main(args)
