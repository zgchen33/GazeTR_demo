import sys
import torch
import cv2
import numpy as np
'''
This script is to test how to use GazeTR
'''

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sys.path.append("./GazeTR_main")
from model import Model

gaze_network = Model().to(device)

gazeTR_weights = torch.load("/home/hxy/few_shot_gaze/gazeTR_demo/GazeTR-H-ETH.pt")
if torch.cuda.device_count() == 1:
    if next(iter(gazeTR_weights.keys())):
        gazeTR_weights = dict([(k, v) for k, v in gazeTR_weights.items()])
gaze_network.load_state_dict(gazeTR_weights)


img = cv2.imread('test_images/right1.jpg')
img_tran = np.transpose(img, [2, 0, 1])
img_tensor = torch.FloatTensor(img_tran[np.newaxis, :, :, :]).to(device)

img_dict = {'face': img_tensor}


# label = torch.ones(10, 2).cuda()
output = gaze_network(img_dict)

print(output)


'''
    the first dimension is right and left
    the second dimension is up and down
right1.jpg:       tensor([[ 0.1126, -0.1595]], device='cuda:0', grad_fn=<AddmmBackward0>)
left1.jpg:     tensor([[-0.2247],[-0.1909]], device='cuda:0', grad_fn=<PermuteBackward0>)
down.jpg:       tensor([[-0.0149, -0.2729]]
down2.jpg:      tensor([[-0.0146, -0.2609]] 
'''


from normalization import draw_gaze

draw_gaze(img, np.array([112, 112]).reshape(2, 1), np.array(1.57, 0).reshape((2, 1)))
cv2.imshow('img', img)
cv2.waitKey(0)