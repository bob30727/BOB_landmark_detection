# https://github.com/thnkim/OpenFacePytorch
# https://github.com/AlfredXiangWu/LightCNN

import json
import os
from xml.dom import minidom as md

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from skimage import io, transform, color, util
from torch.utils.data import Dataset

USE_CUDA = True
USE_PRETRAINED = True
TRAIN_SIDE_FACE = True

CHECKPOINT_MODEL_FILE = "checkpoint.pth.tar"
BEST_MODEL_FILE = "best_model.pth.tar"

class FaceDataDumper(object):
    def dump(self, directory):
        sum_x = np.zeros((3, self.height, self.width))
        sum_x2 = np.zeros((3, self.height, self.width))
        for idx in range(self.__len__()):
            image, landmarks = self.__getitem__(idx)
            sum_x += image
            sum_x2 += image * image
            np.savez(os.path.join(directory, f"face_{idx:04d}.npz"),
                     image=image,
                     landmarks=landmarks)

        mean = 1. * sum_x / self.__len__()
        stddev = np.sqrt(1. * sum_x2  / self.__len__() - mean * mean)
        np.savez(os.path.join(directory, f"normalization.npz"),
                 mean=mean,
                 stddev=stddev)


class GloryLandmarkDataset(Dataset, FaceDataDumper):
    LANDMARK_META = {'Right Brow': 1, 'Right Contour': 9, 'Right Eye': 5, 'Right Lip': 3, 'Right Nose': 3}

    def __init__(self, root_dir, width, height):
        self.images = []
        self.n_landmarks = sum(self.LANDMARK_META.values())
        self.width = width
        self.height = height

        for fname in os.listdir(root_dir):
            if fname.endswith('face_landmark.json'):
                with open(os.path.join(root_dir, fname), "r") as fin:
                    d = json.loads(fin.read())

                path = os.path.join(root_dir, os.path.basename(d['File name']))
                landmarks = []
                for region, n_landmark in self.LANDMARK_META.items():
                    for i in range(n_landmark):
                        landmarks.append(d[region][str(i)])
                self.images.append({
                    'landmarks': landmarks,
                    'path': path})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        print(self.images[idx]['path'])
        image = io.imread(self.images[idx]['path'])
        if len(image.shape) == 2:
            image = color.gray2rgb(image)

        face_top, face_left, face_width, face_height= dlib_face_detector(image, size_threshold=100*100)
        if face_top is None:
            # Face not found, use landmark to define face area
            face_top = min(coor[1] for coor in self.images[idx]['landmarks'])
            face_bottom = max(coor[1] for coor in self.images[idx]['landmarks'])
            face_left = min(coor[0] for coor in self.images[idx]['landmarks'])
            face_right = max(coor[0] for coor in self.images[idx]['landmarks'])
            face_width = face_right - face_left
            face_height = face_bottom - face_top

        image, top, left, width, height = crop_face(image,
                          face_top, face_left, face_width, face_height)
        image = transform.resize(image, (self.height, self.width))

        width_scale = 1.0 * self.width / width
        height_scale = 1.0 * self.height / height

        landmarks = np.ones((self.n_landmarks, 2), dtype=np.float32) * np.nan
        for i in range(self.n_landmarks):
            coor = self.images[idx]['landmarks'][i]
            if coor:
                coor = np.array(coor) - np.array([left, top])
                coor = coor * np.array([width_scale, height_scale])
                landmarks[i, :] = coor
        landmarks = landmarks.reshape((self.n_landmarks * 2,))

        # import matplotlib.pylab as plt
        # _landmarks = landmarks.reshape(21, 2)
        # plt.imshow(image)
        # plt.scatter([landmark[0] for landmark in _landmarks], [landmark[1] for landmark in _landmarks])
        # plt.show()

        # print(image.shape, landmarks.shape)
        return np.rollaxis(image, 2), landmarks


class FaceLandmarksDataset(Dataset, FaceDataDumper):
    """Face Landmarks dataset."""

    def __init__(self, xml_file, root_dir, width, height):
        self.images = []
        self.n_landmarks = 0
        self.width = width
        self.height = height

        for image_xml in md.parse(xml_file).getElementsByTagName("image"):
            path = os.path.join(root_dir, image_xml.getAttribute("file"))
            for box_xml in image_xml.getElementsByTagName("box"):
                top = int(box_xml.getAttribute("top"))
                left = int(box_xml.getAttribute("left"))
                width = int(box_xml.getAttribute("width"))
                height = int(box_xml.getAttribute("height"))
                if top < 0 or left < 0:
                    continue

                landmarks = {}
                for landmark_xml in box_xml.getElementsByTagName("part"):
                    name = int(landmark_xml.getAttribute("name"))
                    x = int(landmark_xml.getAttribute("x"))
                    y = int(landmark_xml.getAttribute("y"))
                    landmarks[name] = [x, y]

                self.n_landmarks = max(self.n_landmarks, max(landmarks.keys()) + 1)
                self.images.append({
                    'top': top,
                    'left': left,
                    'width': width,
                    'height': height,
                    'landmarks': landmarks,
                    'path': path})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = io.imread(self.images[idx]['path'])
        if len(image.shape) == 2:
            image = color.gray2rgb(image)

        image, top, left, width, height = crop_face(image,
                          self.images[idx]['top'], self.images[idx]['left'],
                          self.images[idx]['width'], self.images[idx]['height'])
        image = transform.resize(image, (self.height, self.width))

        width_scale = 1.0 * self.width / width
        height_scale = 1.0 * self.height / height

        landmarks = np.ones((self.n_landmarks, 2), dtype=np.float32) * np.nan
        for i in range(self.n_landmarks):
            coor = self.images[idx]['landmarks'].get(i, None)
            if coor:
                coor = np.array(coor) - np.array([left, top])
                coor = coor * np.array([width_scale, height_scale])
                landmarks[i, :] = coor
        landmarks = landmarks.reshape((self.n_landmarks * 2,))

        # print(image.shape, landmarks.shape)
        return np.rollaxis(image, 2), landmarks



class PreprocessedFaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_list = [ fname for fname in os.listdir(root_dir)
                           if fname.endswith(".npz") and not fname.startswith("normalization")]
        self.data_list.sort()

        normalization = np.load(os.path.join(root_dir, f"normalization.npz"))
        self.image_mean = normalization['mean']
        self.image_stddev = normalization['stddev']

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        _data = np.load(os.path.join(self.root_dir, self.data_list[idx]))
        normalized_image = 1. * (_data['image'] - self.image_mean) / self.image_stddev
        normalized_landmarks = _data['landmarks'] / 227.
        return normalized_image, normalized_landmarks


class LRN(nn.Module):
    def __init__(self, local_size=5, k=2, alpha=0.0001, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x


class TwoLayerFC(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout2d(p=0.3)
        self.conv1 = nn.Conv2d(1, 512, 11, 4)
        self.conv2 = nn.Conv2d(512, 128, 5, 2)
        self.conv3 = nn.Conv2d(128, 64, 3)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 68 * 2)

    def forward(self, x):
        y = self.dropout(x)
        y = functional.max_pool2d(functional.relu(self.conv1(y)), 2)
        y = functional.max_pool2d(functional.relu(self.conv2(y)), 2)
        y = functional.max_pool2d(functional.relu(self.conv3(y)), 2)
        y = y.view(-1, int(np.prod(y.size()[1:])))
        # print(y.size())
        y = functional.relu(self.fc1(y))
        y = self.fc2(y)

        return y


class HyperFace(nn.Module):
    def __init__(self, n_landmark=68):
        super().__init__()
        self.n_landmark = n_landmark

        self.conv1 = nn.Conv2d(3, 96, 11, 4, 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(3, 2)
        # self.norm1 = nn.BatchNorm2d(96)
        self.norm1 = LRN()
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.max2 = nn.MaxPool2d(3, 2)
        # self.norm2 = nn.BatchNorm2d(256)
        self.norm2 = LRN()
        self.conv2a = nn.Conv2d(96, 256, 4, 4)
        self.relu2a = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.relu3 =  nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4a = nn.Conv2d(384, 256, 2, 2)
        self.relu4a = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.max5 = nn.MaxPool2d(3, 2)

        self.conv_a = nn.Conv2d(768, 192, 1)
        self.relu_a = nn.ReLU(inplace=True)
        self.drop_a = nn.Dropout2d(p=0.01)

        self.fc1 = nn.Linear(192 * 6 * 6, 4096)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.drop_fc1 = nn.Dropout(p=0.01)

        self.fc2 = nn.Linear(4096, 1024)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.drop_fc2 = nn.Dropout(p=0.01)

        self.fc3 = nn.Linear(1024, self.n_landmark*2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max1(x)

        y1 = self.conv2a(x)
        y1 = self.relu2a(y1)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        y2 = self.conv4a(x)
        y2 = self.relu4a(y2)
        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        y3 = self.max5(x)

        # print(y1.data.shape, y2.data.shape, y3.data.shape)
        y = torch.cat((y1, y2, y3), 1)
        y = self.conv_a(y)
        y = self.relu_a(y)
        y = self.drop_a(y)
        y = y.view(-1, 6912)
        y = self.fc1(y)
        y = self.relu_fc1(y)
        y = self.drop_fc1(y)
        y = self.fc2(y)
        y = self.relu_fc2(y)
        y = self.drop_fc2(y)
        y = self.fc3(y)

        return y


def dlib_face_detector(image, size_threshold=0):
    import dlib
    detector = dlib.get_frontal_face_detector()

    faces = detector(image, 1)
    if len(faces) == 0:
        print("Face not found by dlib.")
        return None, None, None, None

    face_size = [ (face, (face.bottom() - face.top()) * (face.right() - face.left())) for face in faces]
    face_size.sort(key=lambda x:x[1], reverse=True)
    if face_size[0][1] < size_threshold:
        print("Face size {} too small, giving up!".format(face_size[0][1]))
        return None, None, None, None
    selected_face = face_size[0][0]

    top = selected_face.top()
    left = selected_face.left()
    width = selected_face.right() - selected_face.left()
    height = selected_face.bottom() - selected_face.top()

    return top, left, width, height


def crop_face(image, face_top, face_left, face_width, face_height, buffer_portion = 1./4):
    image_height, image_width, image_channel = image.shape
    face_bottom = face_top + face_height
    face_right = face_left + face_width
    width_buff = int(face_width * buffer_portion)
    height_buff = int(face_height * buffer_portion)
    padding = max(width_buff, height_buff)
    padded_image = util.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'constant')
    top = face_top - height_buff
    bottom = face_bottom + height_buff
    left = face_left - width_buff
    right = face_right + width_buff
    width = right - left
    height = bottom - top

    return padded_image[(padding+top):(padding+bottom), (padding+left):(padding+right), :], top, left, width, height


def predict_face_landmark(image,
                          face_top, face_left, face_width, face_height,
                          model_file_name, normalization_file_name, n_landmark=21):
    target_width = 227
    target_height = 227

    normalization = np.load(normalization_file_name)
    image_mean = normalization['mean']
    image_stddev = normalization['stddev']

    if len(image.shape) == 2:
        image = color.gray2rgb(image)

    _model = torch.load(model_file_name, map_location=lambda storage, loc: storage)
    net = HyperFace(n_landmark)
    net.load_state_dict(_model['state_dict'])

    face_image, top, left, width, height = crop_face(
        image,
        face_top, face_left,
        face_width, face_height,
        buffer_portion=1/4.)

    width_scale = 1. * target_width / width
    height_scale = 1. * target_width / height
    face_image = transform.resize(face_image, (target_height, target_width))
    face_image = np.rollaxis(face_image, 2)
    face_image = 1.0*(face_image - image_mean) / image_stddev
    face_image = face_image[np.newaxis, :, :, :]

    landmarks = net(torch.autograd.Variable(torch.from_numpy(face_image).type(torch.FloatTensor)))
    landmarks = landmarks.data.numpy().reshape((-1, 2))
    landmarks = landmarks / np.array([[width_scale, height_scale]]) * np.array([[target_width, target_height]]) + np.array([left, top])

    return landmarks


def predict(_image, model_file_name, normalization_file_name, n_landmark=21, create_plot=True):
    import dlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    target_width = 227
    target_height = 227

    normalization = np.load(normalization_file_name)
    image_mean = normalization['mean']
    image_stddev = normalization['stddev']

    if type(_image) == str:
        image = io.imread(_image)
    else:
        image = _image

    if create_plot:
        plt.imshow(image)
        ax = plt.gca()

    detector = dlib.get_frontal_face_detector()
    faces = detector(image, 1)

    if len(image.shape) == 2:
        image = color.gray2rgb(image)

    _model = torch.load(model_file_name, map_location=lambda storage, loc: storage)
    net = HyperFace(n_landmark)
    net.load_state_dict(_model['state_dict'])

    all_face_landmarks = []
    for face in faces:
        face_image, top, left, width, height = crop_face(
            image,
            face.top(), face.left(),
            face.right() - face.left(),
            face.bottom() - face.top(),
            buffer_portion=1/4.)
        face_size = (face.right() - face.left()) * (face.bottom() - face.top())

        width_scale = 1. * target_width / width
        height_scale = 1. * target_width / height
        face_image = transform.resize(face_image, (target_height, target_width))
        face_image = np.rollaxis(face_image, 2)
        face_image = 1.0*(face_image - image_mean) / image_stddev
        face_image = face_image[np.newaxis, :, :, :]
        if create_plot:
            ax.add_patch(patches.Rectangle([left, top], width, height, fill=False, color='r'))

        landmarks = net(torch.autograd.Variable(torch.from_numpy(face_image).type(torch.FloatTensor)))
        landmarks = landmarks.data.numpy().reshape((-1, 2))
        # print(landmarks)
        landmarks = landmarks / np.array([[width_scale, height_scale]]) * np.array([[target_width, target_height]]) + np.array([left, top])
        if create_plot:
            plt.scatter([coor[0] for coor in landmarks], [coor[1] for coor in landmarks], s=2)
        all_face_landmarks.append((face_size, landmarks))


    if create_plot:
        plt.show()

    return all_face_landmarks

def plot_data(data_path):
    import matplotlib.pyplot as plt
    _data = np.load(data_path)
    image = np.rollaxis(_data['image'], 0, 3)
    print(image.shape)
    landmarks =  _data['landmarks'].reshape((-1, 2))

    plt.imshow(image)
    plt.scatter([coor[0] for coor in landmarks], [coor[1] for coor in landmarks])
    plt.show()

# preprocessing data
# train_data = FaceLandmarksDataset(
#     # # "/Users/wckao/Documents/Projects/3DMakeup/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml",
#     # "/Users/wckao/Documents/Projects/3DMakeup/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml",
#     # "/Users/wckao/Documents/Projects/3DMakeup/ibug_300W_large_face_landmark_dataset",
#     # "/Users/wckao/Documents/Projects/3DMakeup/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml",
#     # "/home/wckao/Documents/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml",
#     "/home/wckao/Documents/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml",
#     "/home/wckao/Documents/ibug_300W_large_face_landmark_dataset",
#     227, 227
# )
# train_data = GloryLandmarkDataset(
#     # "/Users/wckao/Documents/Projects/3DMakeup/glory_data/Transform",
#     # "/home/wckao/Downloads/Transform_train",
#     "/home/wckao/Downloads/1103/train",
#     227, 227)
# # # train_data.dump("/Users/wckao/Documents/Projects/3DMakeup/deeplandmark/preprocessed_glory")
# train_data.dump("/home/wckao/Documents/deeplandmark/glory_preprocessed_train")


# # code to train NN
# # # net = TwoLayerFC()
# net = HyperFace()
# if USE_CUDA:
#     net.cuda()
#
# criterion = nn.MSELoss()
# optimizer = optim.Adam(net.parameters())
# # train_data = FaceLandmarksDataset(
# #     # "/Users/wckao/Documents/Projects/3DMakeup/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml",
# #     "/Users/wckao/Documents/Projects/3DMakeup/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml",
# #     "/Users/wckao/Documents/Projects/3DMakeup/ibug_300W_large_face_landmark_dataset",
# #     227, 227
# # )
#
# if not TRAIN_SIDE_FACE:
#     train_data = PreprocessedFaceLandmarksDataset(
#         # "/Users/wckao/Documents/Projects/3DMakeup/deeplandmark/preprocessed"
#         "/home/wckao/Documents/deeplandmark/preprocessed"
#     )
#     test_data = PreprocessedFaceLandmarksDataset(
#         # "/Users/wckao/Documents/Projects/3DMakeup/deeplandmark/preprocessed"
#         "/home/wckao/Documents/deeplandmark/preprocessed_test"
#     )
# else:
#     train_data = PreprocessedFaceLandmarksDataset(
#         # "/Users/wckao/Documents/Projects/3DMakeup/deeplandmark/preprocessed_glory"
#         "/home/wckao/Documents/deeplandmark/glory_preprocessed_train"
#     )
#     test_data = PreprocessedFaceLandmarksDataset(
#         # "/Users/wckao/Documents/Projects/3DMakeup/deeplandmark/preprocessed_glory"
#         "/home/wckao/Documents/deeplandmark/glory_preprocessed_test"
#     )
#
# train_loader = DataLoader(
#     train_data,
#     shuffle=True, batch_size=300, num_workers=8)
#
# test_loader = DataLoader(
#     test_data, batch_size=300, num_workers=8)
#
# best_error_rate = np.Inf
#
# if USE_PRETRAINED:
#     try:
#         state = torch.load('checkpoint.pth.tar')
#         best_state = torch.load('best_model.pth.tar')
#     except FileNotFoundError:
#         pass
#     else:
#         net.load_state_dict(state['state_dict'])
#         optimizer.load_state_dict(state['optimizer'])
#         best_error_rate = best_state['test_loss']
#
# if TRAIN_SIDE_FACE:
#     for param in net.parameters():
#         param.requires_grad = False
#
#     layers = ['fc1', 'relu_fc1', 'drop_fc1',
#               'fc2', 'relu_fc2', 'drop_fc2', 'fc3']
#
#     net.fc3 = nn.Linear(1024, 21*2)
#     for layer in layers:
#         for param in net.__getattr__(layer).parameters():
#             param.requires_grad = True
#
#     if USE_CUDA:
#         net.cuda()
#     optimizer = optim.Adam(p for p in net.parameters() if p.requires_grad)
#
#     CHECKPOINT_MODEL_FILE = 'checkpoint_sideface.pth.tar'
#     BEST_MODEL_FILE = 'best_model_sideface.pth.tar'
#     best_error_rate = np.Inf
#
# for epoch in range(2000):
#     running_loss = 0.
#
#     for i, data in enumerate(train_loader):
#         inputs, targets = data
#         inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor)
#         inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
#         if USE_CUDA:
#             inputs, targets = inputs.cuda(), targets.cuda()
#
#         optimizer.zero_grad()
#         output = net(inputs)
#         loss = criterion(output, targets)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.data[0]
#         if i % 10 == 9 or i + 1 == len(train_loader):
#             total_test_loss = 0.
#             for test_i, test_data in enumerate(test_loader):
#                 test_inputs, test_targets = test_data
#                 test_inputs, test_targets = test_inputs.type(torch.FloatTensor), test_targets.type(torch.FloatTensor)
#                 test_inputs, test_targets = torch.autograd.Variable(test_inputs), torch.autograd.Variable(test_targets)
#                 if USE_CUDA:
#                     test_inputs, test_targets = test_inputs.cuda(), test_targets.cuda()
#                 test_output = net(test_inputs)
#                 test_loss = criterion(test_output, test_targets)
#                 total_test_loss += test_loss.data[0]
#
#             print('[%d, %5d] training loss: %.6f testing loss: %.6f' %
#                   (epoch + 1, i + 1, running_loss / (i+1), total_test_loss / len(test_loader)))
#
#             running_loss = 0.
#             if total_test_loss < best_error_rate:
#                 best_error_rate = total_test_loss
#                 torch.save({
#                     'epoch': epoch + 1,
#                     'state_dict': net.state_dict(),
#                     'optimizer': optimizer.state_dict(),
#                     'test_loss': total_test_loss
#                 }, BEST_MODEL_FILE)
#
#
#     torch.save({
#         'epoch': epoch + 1,
#         'state_dict': net.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         'running_loss': running_loss
#     }, CHECKPOINT_MODEL_FILE)
#
#
# print('Finished Training')

# # validate & predict
# predict(
#     # "/home/wckao/Downloads/Transform_train/600_1.jpg",
#     # "/home/wckao/Downloads/Transform_train/rotate_10_618_1.jpg",
#     # "/home/wckao/Downloads/Transform_test/588_1.jpg",
#     # "/home/wckao/Downloads/Transform_test/rotate_10_568_1.jpg",
#     # "/home/wckao/Downloads/Transform_test/rotate_10_588_1.jpg",
#     # "/home/wckao/Downloads/1103/train/rotate_0_575_1.jpg",
#     # "/home/wckao/Downloads/1025/575_1.jpg",
#     # "/home/wckao/Downloads/1025/598_1.jpg",
#     "/home/wckao/Downloads/1025/657_1.jpg",
#     # "/Users/wckao/Documents/Projects/3DMakeup/glory_data/Transform_all/mirror_620_1.jpg",
#     # "/Users/wckao/Documents/Projects/3DMakeup/ibug_300W_large_face_landmark_dataset/ibug/image_084.jpg",
#     # "/Users/wckao/Documents/Projects/3DMakeup/ibug_300W_large_face_landmark_dataset/lfpw/trainset/image_0735.png",
#     # "/Users/wckao/Documents/Projects/3DMakeup/ibug_300W_large_face_landmark_dataset/ibug/image_087.jpg",
#     # "/Users/wckao/Documents/Projects/3DMakeup/deeplandmark/best_model_sideface.pth.tar",
#     # "/Users/wckao/Documents/Projects/3DMakeup/deeplandmark/normalization.npz"
#     "/home/wckao/Documents/deeplandmark/checkpoint_sideface.pth.tar",
#     "/home/wckao/Documents/deeplandmark/glory_preprocessed_train/normalization.npz",
#     21
#
#     # "/home/wckao/Documents/ibug_300W_large_face_landmark_dataset/lfpw/trainset/image_0735.png",
#     # "/home/wckao/Documents/deeplandmark/best_model_lrn.pth.tar",
#     # "/home/wckao/Documents/deeplandmark/preprocessed/normalization.npz",
#     # 68
# )


# for i in range(866):
#     plot_data("/home/wckao/Documents/deeplandmark/glory_preprocessed_train/face_{:04d}.npz".format(i))

