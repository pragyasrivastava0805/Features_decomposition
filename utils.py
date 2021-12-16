import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable




class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img



def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
def mean_accuracy(preds, target):
  num_classes = preds.size(1)
  preds = torch.max(preds, dim=1).indices
  accu_class = []
  for c in range(num_classes):
    mask = (target == c)
    c_count = torch.sum(mask).item()
    if c_count == 0:
      continue
    preds_c = torch.masked_select(preds, mask)
    accu_class += [1.0 * torch.sum(preds_c == c).item() / c_count]
  return 100.0 * np.mean(accu_class)

def transform(train=True):
    OFFICE_MEAN = [0.485, 0.456, 0.406]
    OFFICE_STD = [0.229, 0.224, 0.225]
    resize = 256
    randomResizedCrop = 224
    if train:
        data_transforms = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomResizedCrop(randomResizedCrop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(OFFICE_MEAN, OFFICE_STD)
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(randomResizedCrop),
            transforms.ToTensor(),
            transforms.Normalize(OFFICE_MEAN, OFFICE_STD)
        ])
    return data_transforms
