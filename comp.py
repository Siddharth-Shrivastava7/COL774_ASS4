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
from PIL import Image  
from torchvision.models.inception import BasicConv2d, InceptionA 
from torch.optim.lr_scheduler import StepLR

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import nltk
stopwords = nltk.corpus.stopwords.words('english') 
from nltk.stem import WordNetLemmatizer 
wordnet_lemmatizer = WordNetLemmatizer()


writer = SummaryWriter('comp') 

import argparse 
parser = argparse.ArgumentParser() 
'''
complete the req args 
'''

parser.add_argument('-tr', '--train_tsv_path', default='/home/sidd_s/assign/COL774_ASS4/Train_text.tsv') 
parser.add_argument('-imdirtr', '--image_dir_train', default='/home/sidd_s/assign/data') 
parser.add_argument('-imdirte', '--image_dir_test', default='/home/sidd_s/assign/data')  
parser.add_argument('-ms', '--model_save_path', default='/home/sidd_s/assign/comp.pth')

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
        # img = np.array(image.resize((new_w, new_h), Image.ANTIALIAS)) ## not standardising here
        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))  

        image = torch.tensor(image)  
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
                img_cap_path = os.path.join(args.image_dir_train, img_captions[0])   
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

        global max_len_cap 
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
    
            pad_len = max_len_cap - len(caption.split(' '))  
            caption =  '<st> ' + caption + ' <en>'  + pad_len * ' <pad>' ## padding for ensuring each caption is of same length

            captions_dict[img_p] = caption 

            # max_len_cap += 2
    
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

        # print(lookup_tensor)
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
        image = Image.open(os.path.join(self.img_dir,img_name))
        captions = self.captions_dict[img_name] 
    
        if self.img_transform:
            image = self.img_transform(image)

        if self.captions_transform:
            captions = self.captions_transform(captions)

        # sample = {'image': image, 'captions': captions, 'lengths': lengths}
        sample = {'image': image, 'captions': captions}

        return sample

## Model architecture 

class resnet(nn.Module): 
    def __init__(self, fine_tune_last):
        super().__init__()
        model_resnet = models.resnet18(pretrained=True) 
        modules = list(model_resnet.children())[:-1]  
        self.resnet = nn.Sequential(*modules) 
        self.fine_tune(fine_tune_last)

    def forward(self, x): 
        x = self.resnet(x)  
        return x 

    def fine_tune(self, fine_tune_last, fine_tune_ = True): 
        '''
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        ''' 
        for p in self.resnet.parameters(): 
            p.requires_grad = False 
        ## fine tunning only the latter layers since initial layers capture generic features such as edges, blobs..
        for c in list(self.resnet.children())[:-fine_tune_last]: 
            for p in c.parameters():  
                if fine_tune_:
                    p.requires_grad = True


class OneHot(nn.Module):
    def __init__(self, depth):
        super(OneHot, self).__init__()
        emb = nn.Embedding(depth, depth)
        emb.weight.data = torch.eye(depth)
        emb.weight.requires_grad = False
        self.emb = emb

    def forward(self, input_):
        return self.emb(input_)


## mod attention module 
 
class attention_mod(nn.Module):
    def __init__(self, hidden_size):
        super(attention_mod, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size), requires_grad=True)
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.expand(timestep, -1, -1).transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        return attn_energies.softmax(2)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.expand(encoder_outputs.size(0), -1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy


class decoder(nn.Module): 
    def __init__(self, vocab_size, max_seq_length,  hidden_size, drop_pr=0.1, num_layers=1):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, hidden_size) 
        self.attention = attention_mod(hidden_size) 
        self.rnn = nn.GRU(hidden_size * 2, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, vocab_size)  
        self.max_seq_length = max_seq_length
        self.drop_pr = drop_pr 
        self.hidden_size = hidden_size

    def forward_step(self, input_, last_hidden, encoder_outputs):
        emb = self.emb(input_.transpose(0, 1))
        attn = self.attention(last_hidden, encoder_outputs)
        context = attn.bmm(encoder_outputs).transpose(0, 1)
        rnn_input = torch.cat((emb, context), dim=2)

        outputs, hidden = self.rnn(rnn_input, last_hidden)

        if outputs.requires_grad:
            outputs.register_hook(lambda x: x.clamp(min=-10, max=10))

        # outputs = self.out(outputs.contiguous().squeeze(0)).log_softmax(1)
        outputs = self.out(outputs.contiguous().squeeze(0)) ## logits sending 

        return outputs, hidden

    def forward(self, x, encoder_outputs = None): 
        batch_size = x.shape[0]
        self.batch_size = batch_size
        outputs = [] 
        self.rnn.flatten_parameters() 
        decoder_hidden = torch.zeros(1, batch_size, self.hidden_size, device=encoder_outputs.device)  
    
        for di in range(self.max_seq_length):
            decoder_input = x[:, di].unsqueeze(1)

            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs)

            step_output = decoder_output.squeeze(1)
            outputs.append(step_output)

        outputs = torch.stack(outputs).permute(1, 0, 2) 
        return outputs, decoder_hidden

    def get_bs_pred(self, x, encoder_outputs = None):
        
        outputs = []
        self.rnn.flatten_parameters()  
        decoder_hidden = torch.zeros(1, self.batch_size, self.hidden_size, device=encoder_outputs.device)   

        decoder_input = x[:, 0].unsqueeze(1) ## No teacher forcing 
        for di in range(self.max_seq_length):

            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs)

            step_output = decoder_output.squeeze(1)
            outputs.append(step_output)

        outputs = torch.stack(outputs).permute(1, 0, 2) 
        return outputs, decoder_hidden


class ImageCaptionsNet_mod(nn.Module):
    def __init__(self, img_width, img_height, hidden_size, vocab_size, max_len):
        super().__init__() 

        self.res = resnet(fine_tune_last=2)

        f = self.res(torch.rand(32, 3, img_height, img_width))  
        self._fc = f.size(1)
        self._fh = f.size(2)
        self._fw = f.size(3) 
        self.onehot_x = OneHot(self._fh)
        self.onehot_y = OneHot(self._fw) 
        self.encode_emb = nn.Linear(self._fc + self._fh + self._fw, hidden_size)
        self.decoder = decoder(vocab_size, max_len, hidden_size) 

    def forward(self, input_, target_seq=None):  

        encoder_outputs = self.res(input_) 
        b, fc, fh, fw = encoder_outputs.size() 
        x, y = torch.meshgrid(torch.arange(fh, device=device), torch.arange(fw, device=device))
        h_loc = self.onehot_x(x)
        w_loc = self.onehot_y(y) 
        loc = torch.cat([h_loc, w_loc], dim=2).unsqueeze(0).expand(b, -1, -1, -1)
        encoder_outputs = torch.cat([encoder_outputs.permute(0, 2, 3, 1), loc], dim=3)
        encoder_outputs = encoder_outputs.contiguous().view(b, -1, self._fc + self._fh + self._fw)
        encoder_outputs = self.encode_emb(encoder_outputs) 
        decoder_outputs, decoder_hidden = self.decoder(target_seq, encoder_outputs=encoder_outputs)
        return decoder_outputs
    
    def enc_bs(self, input):
        encoder_outputs = self.res(input) 
        b, fc, fh, fw = encoder_outputs.size() 
        x, y = torch.meshgrid(torch.arange(fh, device=device), torch.arange(fw, device=device))
        h_loc = self.onehot_x(x)
        w_loc = self.onehot_y(y) 
        loc = torch.cat([h_loc, w_loc], dim=2).unsqueeze(0).expand(b, -1, -1, -1)
        encoder_outputs = torch.cat([encoder_outputs.permute(0, 2, 3, 1), loc], dim=3)
        encoder_outputs = encoder_outputs.contiguous().view(b, -1, self._fc + self._fh + self._fw)
        encoder_outputs = self.encode_emb(encoder_outputs) 
        return encoder_outputs


def beam_search_pred(model, x,  decoder, vocab_dict, beam_width=3, log=False):   
    img, cap = x 
    enc_feat = model.enc_bs(img) 
    outputs, hidden = decoder.get_bs_pred(cap, enc_feat)
    
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

    img_transform = transforms.Compose([transforms.Resize(size=(256,256)), transforms.ToTensor(),  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 

    # Set the captions tsv file path
    CAPTIONS_FILE_PATH = args.train_tsv_path
    # print('preprocessing....captions')  
    captions_preprocessing_obj = CaptionsPreprocessing(CAPTIONS_FILE_PATH) 
    vocab = captions_preprocessing_obj.generate_vocabulary()
    vocab_size = len(vocab)

    net = ImageCaptionsNet_mod(img_width=256, img_height=256, hidden_size=2048, vocab_size=vocab_size, max_len=max_len_cap+2) ## 2 adding for end and start token
    net = net.to(device)

    IMAGE_DIR = args.image_dir_train ## train image directory 

    # Creating the Dataset
    dataset = ImageCaptionsDataset(
        IMAGE_DIR, captions_preprocessing_obj.captions_dict, img_transform=img_transform,
        captions_transform=captions_preprocessing_obj.captions_transform
    ) 
    

    # Define your hyperparameters
    NUMBER_OF_EPOCHS = 3000
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 256
    VAL_BATCH_SIZE = 1 
    NUM_WORKERS = 1 # Parallel threads for dataloading
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)   
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

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
                                            sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True) 
    val_loader = DataLoader(dataset, batch_size=VAL_BATCH_SIZE,
                                                    sampler=valid_sampler, num_workers=NUM_WORKERS, pin_memory=True) 

    best_val_loss = np.inf 
    val_check = 1
    for epoch in tqdm(range(NUMBER_OF_EPOCHS)): 
        train_epoch_loss = 0
        for batch_idx, sample in tqdm(enumerate(train_loader)):
            net.zero_grad() 
            net.train() 
            image_batch, captions_batch = sample['image'], sample['captions']

            image_batch, captions_batch = image_batch.to(device, dtype=torch.float), captions_batch.to(device)
    
            output_captions = net(image_batch, captions_batch)  
           
            loss = loss_function(output_captions.reshape(-1, vocab_size), captions_batch.view(-1)) 
            # print(loss)
            train_epoch_loss += loss
            loss.backward()
            optimizer.step()  
            # scheduler.step()

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
                    val_output_captions = net(val_image_batch, val_captions_batch)
                    val_loss = loss_function(val_output_captions.view(-1, vocab_size), val_captions_batch.view(-1)) 
                    # print(val_loss)
                    val_epoch_loss+=val_loss
                val_epoch_loss /= len(val_loader) 
                writer.add_scalar('validation loss', val_epoch_loss.item(), epoch)
                print('val_loss_epoch:',epoch, '  ', val_epoch_loss.item())  
                if val_epoch_loss < best_val_loss: 
                    best_val_loss = val_epoch_loss 
                    torch.save(net.state_dict(), args.model_save_path)
                    print('Model_updated')

    writer.close() 