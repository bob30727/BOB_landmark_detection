from .face_landmarks import HyperFace, PreprocessedFaceLandmarksDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


USE_CUDA = True
USE_PRETRAINED = True
TRAIN_SIDE_FACE = True

CHECKPOINT_MODEL_FILE = "checkpoint.pth.tar"
BEST_MODEL_FILE = "best_model.pth.tar"


# code to train NN
# # net = TwoLayerFC()
net = HyperFace()
if USE_CUDA:
    net.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())
# train_data = FaceLandmarksDataset(
#     # "/Users/wckao/Documents/Projects/3DMakeup/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml",
#     "/Users/wckao/Documents/Projects/3DMakeup/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml",
#     "/Users/wckao/Documents/Projects/3DMakeup/ibug_300W_large_face_landmark_dataset",
#     227, 227
# )

if not TRAIN_SIDE_FACE:
    train_data = PreprocessedFaceLandmarksDataset(
        # "/Users/wckao/Documents/Projects/3DMakeup/deeplandmark/preprocessed"
        "/home/wckao/Documents/deeplandmark/preprocessed"
    )
    test_data = PreprocessedFaceLandmarksDataset(
        # "/Users/wckao/Documents/Projects/3DMakeup/deeplandmark/preprocessed"
        "/home/wckao/Documents/deeplandmark/preprocessed_test"
    )
else:
    train_data = PreprocessedFaceLandmarksDataset(
        # "/Users/wckao/Documents/Projects/3DMakeup/deeplandmark/preprocessed_glory"
        "/home/wckao/Documents/deeplandmark/glory_preprocessed_train"
    )
    test_data = PreprocessedFaceLandmarksDataset(
        # "/Users/wckao/Documents/Projects/3DMakeup/deeplandmark/preprocessed_glory"
        "/home/wckao/Documents/deeplandmark/glory_preprocessed_test"
    )

train_loader = DataLoader(
    train_data,
    shuffle=True, batch_size=300, num_workers=8)

test_loader = DataLoader(
    test_data, batch_size=300, num_workers=8)

best_error_rate = np.Inf

if USE_PRETRAINED:
    try:
        state = torch.load('checkpoint.pth.tar')
        best_state = torch.load('best_model.pth.tar')
    except FileNotFoundError:
        pass
    else:
        net.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        best_error_rate = best_state['test_loss']

if TRAIN_SIDE_FACE:
    for param in net.parameters():
        param.requires_grad = False

    layers = ['fc1', 'relu_fc1', 'drop_fc1',
              'fc2', 'relu_fc2', 'drop_fc2', 'fc3']

    net.fc3 = nn.Linear(1024, 21*2)
    for layer in layers:
        for param in net.__getattr__(layer).parameters():
            param.requires_grad = True

    if USE_CUDA:
        net.cuda()
    optimizer = optim.Adam(p for p in net.parameters() if p.requires_grad)

    CHECKPOINT_MODEL_FILE = 'checkpoint_sideface.pth.tar'
    BEST_MODEL_FILE = 'best_model_sideface.pth.tar'
    best_error_rate = np.Inf

    try:
        state = torch.load('checkpoint_sideface.pth.tar')
        best_state = torch.load('best_model_sideface.pth.tar')
    except FileNotFoundError:
        pass
    else:
        net.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        best_error_rate = best_state['test_loss']


for epoch in range(2000):
    running_loss = 0.

    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        if USE_CUDA:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        output = net(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 10 == 9 or i + 1 == len(train_loader):
            total_test_loss = 0.
            for test_i, test_data in enumerate(test_loader):
                test_inputs, test_targets = test_data
                test_inputs, test_targets = test_inputs.type(torch.FloatTensor), test_targets.type(torch.FloatTensor)
                test_inputs, test_targets = torch.autograd.Variable(test_inputs), torch.autograd.Variable(test_targets)
                if USE_CUDA:
                    test_inputs, test_targets = test_inputs.cuda(), test_targets.cuda()
                test_output = net(test_inputs)
                test_loss = criterion(test_output, test_targets)
                total_test_loss += test_loss.data[0]

            print('[%d, %5d] training loss: %.6f testing loss: %.6f' %
                  (epoch + 1, i + 1, running_loss / (i+1), total_test_loss / len(test_loader)))

            running_loss = 0.
            if total_test_loss < best_error_rate:
                best_error_rate = total_test_loss
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'test_loss': total_test_loss
                }, BEST_MODEL_FILE)


    torch.save({
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'running_loss': running_loss
    }, CHECKPOINT_MODEL_FILE)


print('Finished Training')
