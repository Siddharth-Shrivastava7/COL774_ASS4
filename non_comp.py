from warnings import formatwarning
import torch
from torch._C import dtype
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
import matplotlib.pyplot as plt # for plotting
import numpy as np 
import string 
import os 
from torch.nn.utils.rnn import pack_padded_sequence 
from tqdm import tqdm 


import nltk 
stopwords = nltk.corpus.stopwords.words('english') 
from nltk.stem import WordNetLemmatizer 
wordnet_lemmatizer = WordNetLemmatizer()


import argparse 

parser = argparse.ArgumentParser() 
'''
complete the req args 
'''

parser.add_argument('-tr', '--train_tsv_path', default='/home/cse/phd/anz208849/assignments/COL774/ass4/Train_text.tsv') 
parser.add_argument('-imdirtr', '--image_dir_train', default='/home/cse/phd/anz208849/assignments/COL774/ass4/train_data') 
parser.add_argument('-imdirte', '--image_dir_test', default='/home/cse/phd/anz208849/assignments/COL774/ass4/test_data') 

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w) 
        # print(image)
        img = transform.resize(image, (new_h, new_w))  ## pixel value are between 0 and 1 
        # print(img)
        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))  

        image = torch.tensor(image)  

        # print(image.shape)    
        # print(image)
        return image


class CaptionsPreprocessing:
    """Preprocess the captions, generate vocabulary and convert words to tensor tokens

    Args:
        captions_file_path (string): captions tsv file path
    """
    def __init__(self, captions_file_path):
        self.captions_file_path = captions_file_path

        # Read raw captions
        self.raw_captions_dict = self.read_raw_captions()

        # Preprocess captions
        self.captions_dict = self.process_captions()

        # Create vocabulary
        self.vocab = self.generate_vocabulary()

    def read_raw_captions(self):
        """
        Returns:
            Dictionary with raw captions list keyed by image ids (integers)
        """
        captions_dict = {}
        with open(self.captions_file_path, 'r', encoding='utf-8') as f:
            for img_caption_line in f.readlines():
                img_captions = img_caption_line.strip().split('\t')
                captions_dict[img_captions[0]] = img_captions[1]
                # print(img_captions[0])
                # break
        # print(captions_dict)
        return captions_dict

    def process_captions(self):
        """
        Use this function to generate dictionary and other preprocessing on captions
        """
        raw_captions_dict = self.raw_captions_dict
        # Do the preprocessing here
        captions_dict = raw_captions_dict
        ## complete the for preprocessing here 

        max_len_cap = 0 
        # print(captions_dict)
        ## max len caption find and save the value for making each caption of equal length 
        for caption in captions_dict.values(): 
            # print(caption)
            len_cap = len(caption.split(' '))  
            # print(len_cap)
            if max_len_cap < len_cap: 
                max_len_cap = len_cap 
            # print(max_len_cap)
        
        for img_p, caption in captions_dict.items():  
            # print(caption)
            caption=" ".join([i for i in caption.split() if i not in string.punctuation]) 
            caption = caption.lower()  
            # print(caption)
            # remove single letter words such as a .. s 
            caption = " ".join([word for word in caption.split() if len(word)>1])
            # removing numbers from caption (optional) 
            # caption = " ".join([word for word in caption.split() if word.isalpha()])  
            ## remove stop words  
            caption = " ".join([word for word in caption.split() if word not in stopwords])
            ##  performing lemmatization over stemming (for meaningful words generations with the short form) 
            caption = " ".join([wordnet_lemmatizer.lemmatize(word) for word in caption.split()])  
            # print(caption) 

            pad_len = max_len_cap - len(caption.split(' '))  
            # print(pad_len)
            # print(caption)

            caption =  '<st> ' + caption + ' <en>'  + pad_len * ' <pad>' ## padding for ensuring each caption is of same length

            captions_dict[img_p] = caption
    
        return captions_dict

    def generate_vocabulary(self):
        """
        Use this function to generate dictionary and other preprocessing on captions
        """
        captions_dict = self.captions_dict
        # Generate the vocabulary 
        # print(captions_dict)

        vocab_threshold = 6 ## hyperparam 

        train_captions = []
        for idx,cap in self.captions_dict.items():   
            train_captions.append(cap) 

        # print(train_captions)
        
        word_counts = {} 
        for cap in train_captions: 
            for word in cap.split(' '):
                word_counts[word] = word_counts.get(word, 0) + 1 

        vocab = [w for w in word_counts if word_counts[w] >= vocab_threshold] 

        ## unknown word/token if encountered i.e. word in caption but not in vocab 
        vocab.append('<unkn>')
        # print(vocab)

        return vocab
    
    def captions_transform(self, img_caption_list):
        """
        Use this function to generate tensor tokens for the text captions
        Args:
            img_caption_list: List of captions for a particular image
        """
        vocab = self.vocab 
        # Generate tensors

        word_map = dict(zip(iter(vocab), range(len(vocab))))
        img_capt_lst_tensor = []
        
        # print(word_map)
        # print(img_caption_list)

        ## word is a token 
        # for caption in img_caption_list: 
        #     # print(caption)

        # lookup_tensor = torch.tensor([word_map[word] for word in img_caption_list.split() if word in vocab]) ## will produce different size tensor which is not useful
        # img_capt_lst_tensor.append(lookup) 

        lookup_tensor = torch.tensor([word_map[word] if word in vocab else word_map['<unkn>'] for word in img_caption_list.split()])

        # print(lookup_tensor)
        return lookup_tensor
        # return torch.zeros(len(img_caption_list), 10)

## dataset class 

class ImageCaptionsDataset(Dataset):

    def __init__(self, img_dir, captions_dict, img_transform=None, captions_transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            captions_dict: Dictionary with captions list keyed by image paths (strings)
            img_transform (callable, optional): Optional transform to be applied
                on the image sample.

            captions_transform: (callable, optional): Optional transform to be applied
                on the caption sample (list).
        """
        self.img_dir = img_dir
        self.captions_dict = captions_dict
        self.img_transform = img_transform
        self.captions_transform = captions_transform

        self.image_ids = list(captions_dict.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx] 
        # image = io.imread(os.path.join(self.img_dir,img_name)) 
        image = io.imread(img_name)
        captions = self.captions_dict[img_name] 
        # print(captions) 
        # print('**************')
        # print(len(captions.split(' ')))
        # lengths = len(captions.split(' '))

        if self.img_transform:
            image = self.img_transform(image)

        if self.captions_transform:
            captions = self.captions_transform(captions)

        # print(captions)

        # sample = {'image': image, 'captions': captions, 'lengths': lengths}
        sample = {'image': image, 'captions': captions}

        return sample

## Model architecture 

class ImageCaptionsNet(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, drop_pr=0.3, num_layers=1, fine_tune_start = 5, max_seq_length =20):
        super(ImageCaptionsNet, self).__init__()

        ## CNN Encoder 
        # resnet = models.resnet152(pretrained=True) 
        resnet = models.resnet101(pretrained=True)  
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size) 
        # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)   ## to try 
        ## tunning only few layers since the dataset is small
        self.fine_tune(fine_tune_start)

        ## LSTM decoder 
        # self.drop_prob= drop_pr 
        self.lstm = nn.LSTM(embed_size, hidden_size , num_layers, batch_first=True) 
        self.dropout = nn.Dropout(drop_pr)   
        self.embed = nn.Embedding(vocab_size, embed_size)   
        self.linear_lstm = nn.Linear(hidden_size, vocab_size)
        # self.max_seq_length = max_seq_length   

        # Define your architecture here
        

    def forward(self, x):
    # def forward(self, x):
        image_batch, captions_batch  = x 
        # image_batch, captions_batch, lengths  = x 

        # Forward Propogation 
        ## cnn encoder 
        feat = self.resnet(image_batch) 
        feat = feat.reshape(feat.size(0), -1) 
        # feat = self.bn(self.linear(feat))
        feat = self.linear(feat)  

        # print(feat.shape)  # torch.Size([32, 256]) 

        ## lstm decoder 
        embeddings = self.embed(captions_batch) 
        # print(embeddings.shape)  # torch.Size([32, 10, 256])
        # print(feat.unsqueeze(1).shape) # torch.Size([32, 1, 256])
        embeddings = torch.cat((feat.unsqueeze(1), embeddings[:, :-1,:]), dim=1)   
        # print(embeddings.shape) # torch.Size([32, 10, 256]) 
        # embeddings = torch.cat((feat.unsqueeze(1), embeddings), dim=1) 
        embeddings = self.dropout(embeddings)
        # print(embeddings.shape) 
        # packed = pack_padded_sequence(embeddings, lengths, batch_first=True)  ## not knowing its significance currently  
        # print(packed.shape)
        hiddens, c = self.lstm(embeddings) 
        # print(hiddens.shape) # torch.Size([32, 10, 256]) 
        outputs = self.linear_lstm(hiddens)  
        # print(outputs.shape) # torch.Size([32, 10, 1837])
        # hiddens, c = self.lstm(packed) 
        # outputs = self.linear_lstm(hiddens[0]) 
        # print(outputs) 

        return outputs


    def fine_tune(self, fine_tune_ = True, fine_tune_start = 5): 
        '''
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        ''' 
        for p in self.resnet.parameters(): 
            p.requires_grad = False 
        ## fine tunning only the latter layers since initial layers capture generic features such as edges, blobs..
        for c in list(self.resnet.children())[fine_tune_start:]: 
            for p in c.parameters():  
                if fine_tune_:
                    p.requires_grad = True



def sample_beam_search(features, states=None): 
    pass 

def blue_score(refernec, candidate):  
    pass 



if __name__ == '__main__': 

    IMAGE_RESIZE = (256, 256)
    # Sequentially compose the transforms ## normalising the image acc to  imagenet mean and std (so that when pretrained model of imagenet could be useful)
    img_transform = transforms.Compose([Rescale(IMAGE_RESIZE), ToTensor(),  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Set the captions tsv file path
    CAPTIONS_FILE_PATH = args.train_tsv_path
    # print('preprocessing....captions')  
    captions_preprocessing_obj = CaptionsPreprocessing(CAPTIONS_FILE_PATH) 
    vocab = captions_preprocessing_obj.generate_vocabulary()
    vocab_size = len(vocab)

    # print(vocab_size) # 1837 ## for the current model 

    net = ImageCaptionsNet(embed_size=256, hidden_size=256, vocab_size=vocab_size)
    # If GPU training is required
    # net = net.cuda() 
    net = net.to(device)

    IMAGE_DIR = args.image_dir_train ## train image directory 
    # Creating the Dataset
    train_dataset = ImageCaptionsDataset(
        IMAGE_DIR, captions_preprocessing_obj.captions_dict, img_transform=img_transform,
        captions_transform=captions_preprocessing_obj.captions_transform
    ) 
    # Define your hyperparameters
    NUMBER_OF_EPOCHS = 3
    LEARNING_RATE = 1e-1
    BATCH_SIZE = 32
    NUM_WORKERS = 0 # Parallel threads for dataloading
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE) 

    # Creating the DataLoader for batching purposes
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)  
    # print(train_loader)

    import os
    for epoch in tqdm(range(NUMBER_OF_EPOCHS)):
        for batch_idx, sample in tqdm(enumerate(train_loader)):
            net.zero_grad()  
            image_batch, captions_batch = sample['image'], sample['captions']
            # image_batch, captions_batch, lengths_batch = sample['image'], sample['captions'], sample['lengths'] 

            # captions_batch = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted =False)[0] 

            # If GPU training required
            # image_batch, captions_batch = image_batch.cuda(), captions_batch.cuda() 
            image_batch, captions_batch = image_batch.to(device, dtype=torch.float), captions_batch.to(device)
            # image_batch, captions_batch, lengths_batch = image_batch.to(device, dtype=torch.float), captions_batch.to(device), lengths_batch.to(device)

            output_captions = net((image_batch, captions_batch)) 
            # print(output_captions.shape) # torch.Size([32, 10, 1837])
            # output_captions = net((image_batch, captions_batch, lengths_batch)) 
            # print(captions_batch.shape) # torch.Size([32, 10])
            loss = loss_function(output_captions.view(-1, vocab_size), captions_batch.view(-1)) 
            print(loss.item())
            loss.backward()
            optimizer.step()
    
        print("Iteration: " + str(epoch + 1))



    
