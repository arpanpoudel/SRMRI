from pathlib import Path
from glob import glob
from pathlib import Path
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
#from util import normalize_np

#MRI dataset
#MRI dataset
class MRI_dataset(Dataset):
    "Dataset class for MRI data"
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.lr_data_list=list(self.root.glob('LR/*.npy'))
        self.transform = transform
    
    
    def load_image(self, file_path):
        image = np.load(file_path).astype(np.float32)
        # Apply min-max normalization
        #image = normalize_np(image)  #already normalized
        return image
    
    def get_hr_path(self, lr_path):
        parent_dir = lr_path.parent.parent
        base_name = lr_path.stem
        hr_path = parent_dir / f'HR/{base_name}.npy'
        return hr_path
    
    def __len__(self):
        return len(self.lr_data_list)
    
    def __getitem__(self, index):
        file_path= self.lr_data_list[index]
        lr = self.load_image(file_path)
        path_hr=self.get_hr_path(file_path)
        hr=self.load_image(path_hr)
        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)
        sample = {'lr': lr, 'hr': hr}
        return sample
      
    
def create_dataloader(configs,evaluation=False,transform=None):
    shuffle =True if not evaluation else False
    if transform is None:
        transform = T.Compose([T.ToTensor()])
    train_dataset = MRI_dataset(configs.data.train,transform=transform)
    eval_dataset = MRI_dataset(configs.data.eval,transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.training.batch_size, shuffle=shuffle,drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=configs.training.batch_size, shuffle=shuffle,drop_last=True)
    return train_dataloader,eval_dataloader