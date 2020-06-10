import os
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader

data_cat = ['train', 'valid'] # data categories

def tile_patriot(img, sz=128, N=16):
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                 constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    return img#[N,size,size,3]


# Customized Dataset
class ProstateCancerDataset(Dataset):
    """Prostate Cancer Biopsy Dataset"""
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file
            root_dir (string): Path to the directory with all images
            transform (callable, optional): Optional transform to be applied on an image sample
        """
        # Shuffle dataframes with fixed seed; otherwise, validation set only get cancerous samples
        self.cancer_df = pd.read_csv(csv_file).sample(frac=1, random_state=1)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.cancer_df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, f'{self.cancer_df.iloc[idx, 0]}_0.png')
        # (D,W,H)
        img = io.imread(img_path)
        image = tile_patriot(img, sz=128, N=16) # added new 
        img = cv2.hconcat([cv2.vconcat([image[0], image[1], image[2], image[3]]), 
                             cv2.vconcat([image[4], image[5], image[6], image[7]]), 
                             cv2.vconcat([image[8], image[9], image[10], image[11]]), 
                             cv2.vconcat([image[12], image[13], image[14], image[15]])])
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        isup = self.cancer_df.iloc[idx, 2]
        gleason = self.cancer_df.iloc[idx, 3]
        
        if self.transform:
            img = self.transform(img)
        sample = {'image': img, 'isup_grade': isup, 'gleason_score': gleason}
        return sample      

def get_dataloaders(data, batch_size=8):
    '''
    Returns dataloader pipeline with data augmentation
    '''
    data_transforms = {
        'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.90960454,0.81946206,0.87811487], [0.13244118,0.24944844,0.16392948]) 
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.90960454,0.81946206,0.87811487], [0.13244118,0.24944844,0.16392948])
        ]),
    }
    image_datasets = {x: ProstateCancerDataset(data[x], transform=data_transforms[x]) for x in data_cat}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in data_cat}
    return dataloaders

if __name__=='main':
    pass