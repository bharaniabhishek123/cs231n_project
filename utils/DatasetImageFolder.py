
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, sampler, Subset, WeightedRandomSampler

import torch
from torchvision import datasets

"""
Created from Image Folder and performing save operation for every image it pre-processes. 

In one epoch, it will save all the images and then we can use the saved images later to actual training. This will save pre-processing image everytime

"""


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
       
    @classmethod
    def preprocess(cls, image):
        WINDOW_SIZE = 128
        STRIDE = 64
        K = 16
        
        image, best_coordinates, best_regions = generate_patches(image, window_size=WINDOW_SIZE, stride=STRIDE, k=K)
        glued_image = glue_to_one_picture(best_regions, window_size=WINDOW_SIZE, k=K)
        
        return glued_image           
    
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        image, isup_grade = np.array(original_tuple[0]), original_tuple[1]
        image = self.preprocess(image)
        PIL_image = Image.fromarray((image * 255).astype(np.uint8))

        # Let's get the image file path and extract file_name
        path = self.imgs[index][0]
        file_name = path.split("/")[-1].replace(".jpg", "_g.jpg")
        PIL_image.save(file_name)
        
        # Apply Tranformation
        transform=Compose([Resize((224,224)),ToTensor(), 
                           Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        
        image = transform(PIL_image)

        return {'image': image, 'isup_grade' : isup_grade}  



"""
## Dataset II
Created from ImageFolder on big size .tiff files. This dataset class does not include save operation

"""


class BiopsyDataClass(Dataset):
    
    def __init__(self, image_path, transform=None):
        super(BiopsyDataClass, self).__init__()
        self.data = datasets.ImageFolder(image_path)    # Create data from folder
        
        self.transform = transform
        
    @classmethod
    def preprocess(cls, image):
        WINDOW_SIZE = 128
        STRIDE = 64
        K = 16
        
        image, best_coordinates, best_regions = generate_patches(image, window_size=WINDOW_SIZE, stride=STRIDE, k=K)
        glued_image = glue_to_one_picture(best_regions, window_size=WINDOW_SIZE, k=K)
        
        return glued_image        
        
    def __getitem__(self, idx):

        image, isup_grade = self.data[idx]
        image = np.array(image)
        image = self.preprocess(image)
        PIL_image = Image.fromarray((image * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(PIL_image) 
#             Image.save()
#             print(PIL_image.type)
        return {'image': image, 'isup_grade' : isup_grade}        
    
    def __len__(self):
        return len(self.data)




"""
Created from csv and then do a lookup for image_id inside train_images folder. Before using this filter the csv to exclude the image_id not present in train_images folder.
"""
class BasicDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """        
        self.data_frame = pd.read_csv(csv_file)

        ## Local images only  
        
        # local_files_df = DataFrame([os.path.splitext(filename)[0] for filename in os.listdir('/Users/abharani/Documents/myworkspace/cs231n_project/data/train_images/') if filename.endswith(".tiff")], columns=["image_id"])
        # self.data_frame = local_files_df.join(data_frame.set_index('image_id'), on='image_id')
        # ##

        self.data_dir = data_dir
        self.transform = transform
        logging.info('Creating dataset with {} examples'.format(len(self.data_frame)))
        
    def __len__(self):
        return len(self.data_frame)

    @classmethod
    def preprocess(cls, image):
        WINDOW_SIZE = 128
        STRIDE = 64
        K = 16
        
        image, best_coordinates, best_regions = generate_patches(image, window_size=WINDOW_SIZE, stride=STRIDE, k=K)
        glued_image = glue_to_one_picture(best_regions, window_size=WINDOW_SIZE, k=K)
        
        return glued_image


    def __getitem__(self, idx):
        
        image_id = self.data_frame.iloc[idx]['image_id']
        image_file_path = os.path.join(os.path.join(self.data_dir,'train_images'),image_id + ".tiff")
        image = skimage.io.MultiImage(image_file_path)[-1]
        image = np.array(image)

        isup_grade = self.data_frame.iloc[idx]['isup_grade']

        image = self.preprocess(image)
        PIL_image = Image.fromarray((image * 255).astype(np.uint8))
#         print("Image type {}".format(image.type)) #numpy.ndarray
#         print(PIL_image.type)                     # Image object      

        
        if self.transform:
            
            image = self.transform(PIL_image)
        
        return {'image': image, 'isup_grade' : isup_grade}


