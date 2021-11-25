import torch
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
from torch.utils.data.sampler import SubsetRandomSampler
import os 
import math 
from tensorboardX import SummaryWriter 
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


torch.backends.cudnn.benchmark = True 

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import nltk
stopwords = nltk.corpus.stopwords.words('english') 
from nltk.stem import WordNetLemmatizer 
wordnet_lemmatizer = WordNetLemmatizer()


writer = SummaryWriter() 

import argparse 
parser = argparse.ArgumentParser() 
'''
complete the req args 
'''

parser.add_argument('-tr', '--train_tsv_path', default='/home/sidd_s/assign/COL774_ASS4/Train_text.tsv') 
parser.add_argument('-imdirtr', '--image_dir_train', default='/home/sidd_s/assign/data') 
parser.add_argument('-imdirte', '--image_dir_test', default='/home/sidd_s/assign/data')  
parser.add_argument('-ms', '--model_save_path', default='/home/sidd_s/assign/non_comp.pth')

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
                # print(img_caption_line) 
                img_captions = img_caption_line.strip().split('\t') 
                img_cap_path = os.path.join(args.image_dir_train, img_captions[0])   
                # print(img_cap_path)
                captions_dict[img_captions[0]] = img_captions[1]
                
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
        

        lookup_tensor = torch.tensor([word_map[word] if word in vocab else word_map['<unkn>'] for word in img_caption_list.split()])

        return lookup_tensor


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
        image = io.imread(os.path.join(self.img_dir,img_name)) 
        captions = self.captions_dict[img_name] 
    
        if self.img_transform:
            image = self.img_transform(image)

        if self.captions_transform:
            captions = self.captions_transform(captions)

        sample = {'image': image, 'captions': captions}

        return sample

## Model architecture 

class ImageCaptionsNet(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, drop_pr=0.3, num_layers=1, fine_tune_start = 5, max_seq_length =20):
        super(ImageCaptionsNet, self).__init__()

        ## CNN Encoder 
        resnet = models.resnet101(pretrained=True)  
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size) 
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)   ## to try  
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
        
        ## cnn encoder 
        feat = self.resnet(image_batch) 
        feat = feat.reshape(feat.size(0), -1) 
        feat = self.bn(self.linear(feat))  ## adding batch norm  


        ## lstm decoder 
        embeddings = self.embed(captions_batch) 
        embeddings = torch.cat((feat.unsqueeze(1), embeddings[:, :-1,:]), dim=1)   
        hiddens, c = self.lstm(embeddings) 
        outputs = self.linear_lstm(hiddens)  
        return outputs, feat


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



def beam_search_pred(model, image,  vocab_dict, beam_width=3, log=False): 
    
    

    pass  




def blue_score(references, candidates, sentence = False):  
    if sentence:
        ## candidates: list of tokens
        ## references: list of tokens (can be multiple for one doc)
        score = sentence_bleu(references, candidates)  
    else:
        ## candidates: list of list of tokens
        ## references: list of doc where each doc is a list of references which is list of tokens
        score = corpus_bleu(references, candidates)  
    return score


if __name__ == '__main__': 

    IMAGE_RESIZE = (256, 256)
    # Sequentially compose the transforms ## normalising the image acc to  imagenet mean and std (so that when pretrained model of imagenet could be useful)
    img_transform = transforms.Compose([Rescale(IMAGE_RESIZE), ToTensor(),  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


    CAPTIONS_FILE_PATH = args.train_tsv_path
 
    captions_preprocessing_obj = CaptionsPreprocessing(CAPTIONS_FILE_PATH) 
    vocab = captions_preprocessing_obj.generate_vocabulary()
    vocab_size = len(vocab)


    net = ImageCaptionsNet(embed_size=512, hidden_size=512, vocab_size=vocab_size)

    net = net.to(device)

    IMAGE_DIR = args.image_dir_train ## train image directory 


    dataset = ImageCaptionsDataset(
        IMAGE_DIR, captions_preprocessing_obj.captions_dict, img_transform=img_transform,
        captions_transform=captions_preprocessing_obj.captions_transform
    ) 

    NUMBER_OF_EPOCHS = 3000
    LEARNING_RATE = 1e-1
    BATCH_SIZE = 32
    VAL_BATCH_SIZE = 1 
    NUM_WORKERS = 0 # Parallel threads for dataloading
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE) 

    validation_split = 0.1
    dataset_size = len(dataset) 
    indices = list(range(dataset_size)) 
    split = int(np.floor(validation_split * dataset_size)) 
    shuffle_dataset = True
    if shuffle_dataset:
        np.random.seed(14)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]  
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, 
                                            sampler=train_sampler, num_workers=NUM_WORKERS) 
    val_loader = DataLoader(dataset, batch_size=VAL_BATCH_SIZE,
                                                    sampler=valid_sampler, num_workers=NUM_WORKERS) 
 


    best_val_loss = np.inf 
    val_check = 1
    for epoch in tqdm(range(NUMBER_OF_EPOCHS)): 
        train_epoch_loss = 0
        for batch_idx, sample in tqdm(enumerate(train_loader)):
            net.zero_grad() 
            net.train() 
            image_batch, captions_batch = sample['image'], sample['captions']
        
            image_batch, captions_batch = image_batch.to(device, dtype=torch.float), captions_batch.to(device)
        

            output_captions,_ = net((image_batch, captions_batch))  
            loss = loss_function(output_captions.view(-1, vocab_size), captions_batch.view(-1))  
            train_epoch_loss += loss
            loss.backward()
            optimizer.step()  


        train_epoch_loss /= len(train_loader)
        print('train_loss_epoch:',epoch, '  ', train_epoch_loss.item())
        writer.add_scalar('training loss', train_epoch_loss.item(), epoch) 

        if epoch % val_check == 0: 
            with torch.no_grad():
                net.eval()
                val_epoch_loss = 0 
                print('in_validation')
                for val_sample in tqdm(val_loader):
                    val_image_batch, val_captions_batch = val_sample['image'], val_sample['captions']
                    val_image_batch, val_captions_batch = val_image_batch.to(device, dtype=torch.float), val_captions_batch.to(device)
                    val_output_captions,_ = net((val_image_batch, val_captions_batch))
                    val_loss = loss_function(val_output_captions.view(-1, vocab_size), val_captions_batch.view(-1)) 
                    val_epoch_loss+=val_loss
                val_epoch_loss /= len(val_loader) 
                writer.add_scalar('validation loss', val_epoch_loss.item(), epoch)
                print('val_loss_epoch:',epoch, '  ', val_epoch_loss.item())  
                if val_epoch_loss < best_val_loss: 
                    best_val_loss = val_epoch_loss 
                    torch.save(net.state_dict(), args.model_save_path)
                    print('Model_updated')

    writer.close() 



    
