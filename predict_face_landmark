from .face_landmarks import predict
import sys
import warnings

BasePath = "/home/wckao/Documents/deeplandmark/"
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    all_face_landmarks = predict(sys.argv[1],
                        BasePath + "best_model_sideface.pth.tar",
                        BasePath + "normalization.npz", 21, False)

all_face_landmarks.sort(key=lambda x:x[0], reverse=True)
for coor in all_face_landmarks[0][1]:
    print("{} {}".format(coor[0], coor[1]))
