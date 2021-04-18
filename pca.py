import torch

PATH = ""

state_dict = torch.load(PATH)

state_dict["decoder.fc"]