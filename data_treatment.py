### Align eyes and mouth with fixed points for all images

# %% Imports & installs
from PIL import Image
import face_recognition
from torchvision import datasets, transforms
import glob
import torch

# %% Init values

image_size = 128
batch_size = 64
PATH = ".\\images\\"
test_PATH = ".\\subset\\"

# %% Get average facial landmark positions
"""  NOT ACTUALLY NEEDED, cf next cells

def landmark_position(positions): #center of mass
    X, Y = 0, 0
    for (x, y) in positions:
        X += x
        Y += y
    ln = len(positions)
    return (X//ln, Y//ln)


right_eye = []
left_eye = []
bottom_lip = []
valid_count = 0 
for img_path in glob.glob(PATH + '*.png'):
    img = face_recognition.load_image_file(img_path)
    landmarks = face_recognition.face_landmarks(img)
    if len(landmarks) == 1: #Check if there is only 1 person
        valid_count += 1
        landmarks=landmarks[0]
        #print(landmarks.keys())
        right_eye.append(landmark_position(landmarks["right_eye"]))
        left_eye.append(landmark_position(landmarks["left_eye"]))
        bottom_lip.append(landmark_position(landmarks["bottom_lip"]))

avg_right_eye = landmark_position(right_eye)
avg_left_eye = landmark_position(left_eye)
avg_bottom_lip = landmark_position(bottom_lip)

"""
# %%
#print(f"Average eyes : {avg_left_eye}, {avg_right_eye}, lip : {avg_bottom_lip}")
# %%
#right_eye[:20]
# %%
#bottom_lip[:20]
# On remarque que les données sont déjà centrées

# %%

dataset = datasets.ImageFolder(PATH, transform = transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                        shuffle=True)
testset = datasets.ImageFolder(test_PATH, transform = transforms.ToTensor())

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                        shuffle=True)