import os
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image



class Nuclie_data(Dataset):
    def __init__(self, path, img_size=256, is_train_mode=True):
        self.path = path
        self.img_size = img_size
        self.folders = os.listdir(path)
        self.is_train_mode = is_train_mode
        # self.transforms = get_transforms(0.5, 0.5)
        if self.is_train_mode:
            print('Using Augmentation')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])

        else:
            print('No Augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

            self.gt_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()])

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        image_folder = os.path.join(self.path, self.folders[idx], 'images/')
        mask_folder = os.path.join(self.path, self.folders[idx], 'masks/')
        image_path = os.path.join(image_folder, os.listdir(image_folder)[0])

        img = io.imread(image_path)[:, :, :3].astype('float32')
        img = Image.fromarray(img, mode='RGB')
        mask = self.get_mask(mask_folder, self.img_size, self.img_size)
        mask = Image.fromarray(mask, mode='L')
        img = self.img_transform(img)
        mask = self.gt_transform(mask)
        mask = mask.squeeze(0)
        return img, mask.long()

    def get_mask(self, mask_folder, IMG_HEIGHT, IMG_WIDTH):
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
        for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(os.path.join(mask_folder, mask_))
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask_ = np.expand_dims(mask_, axis=-1)
            mask = np.maximum(mask, mask_)

        mask = (mask * 255).astype('uint8').squeeze()
        return mask



def build_dataset(args):
    data_path = args.data_root + '/stage1_train'
    total_dataset = Nuclie_data(data_path)
    train_set, valid_set = random_split(total_dataset, [580, 90])
    return train_set, valid_set


if __name__ == '__main__':
    base_dir = '/mnt/d/DataScienceBowl2018'
    train_dir = base_dir + '/stage1_train'
    test_dir = base_dir + '/stage1_test'
    train_data = Nuclie_data(train_dir)
    test_data = Nuclie_data(test_dir, is_train_mode=False)
    print(train_data.__getitem__(0))
