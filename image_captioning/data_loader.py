import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: feature directory.  /VIST/resnet_feature
            json: coco annotation file path.   dii-isolation
            vocab: vocabulary wrapper.  
            transform: image transformer.
        """
        self.root = root
        self.json_data = json  # photoflickrids dict.keys keys is the flickr id and baki in the value
        self.ids = list(self.json_data.keys()) # photoflickrids dict.keys keys is the flickr id and baki in the value
        self.vocab = vocab
        # self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        # coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = self.json_data[ann_id] # text
        path = ann_id + '.npy' 
        feature_path = os.path.join(self.root,path)
        feature=np.load(feature_path)
        feature_tensor = torch.from_numpy(feature)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        return feature_tensor, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    feature, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    feature = torch.stack(feature, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]

    targets = torch.zeros(len(captions), max(lengths)).long()
    
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]    
            
    return feature, targets, lengths

def get_loader(root, json, vocab, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,  
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader