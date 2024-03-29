from nltk.util import pr
import torch
from torch import nuclear_norm
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
from Beam_search_comp import beam_search_pred, greedy_search

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import nltk
stopwords = nltk.corpus.stopwords.words('english') 
from nltk.stem import WordNetLemmatizer 
wordnet_lemmatizer = WordNetLemmatizer()


writer = SummaryWriter('comp_again') 

import argparse 
parser = argparse.ArgumentParser() 
'''
complete the req args 
'''

parser.add_argument('-tr', '--train_tsv_path', default='/home/sidd_s/assign/COL774_ASS4/Train_text.tsv') 
parser.add_argument('-imdirtr', '--image_dir_train', default='/home/sidd_s/assign/data') 
parser.add_argument('-imdirte', '--image_dir_test', default='/home/sidd_s/assign/data/test_data')  
parser.add_argument('-ms', '--model_save_path', default='/home/sidd_s/assign/comp_again.pth') 
parser.add_argument('-md', '--model_load_path', default='/home/sidd_s/assign/comp_again.pth')  
parser.add_argument('--train', action='store_true')   
parser.add_argument('--test', action='store_true')   
parser.add_argument('--val', action='store_true') 

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
        
        max_len_cap += 2  
        for img_p, caption in captions_dict.items():  
    
            # pad_len = max_len_cap - len(caption.split(' '))  
            # caption =  '<st> ' + caption + ' <en>'  + pad_len * ' <pad>' ## padding for ensuring each caption is of same length
            caption =  '<st> ' + caption + ' <en>'  
            # print(caption)

            captions_dict[img_p] = caption 

        return captions_dict

    def generate_vocabulary(self):
        """
        Use this function to generate dictionary and other preprocessing on captions
        """
        captions_dict = self.captions_dict
        # Generate the vocabulary 
        # print(captions_dict)

        vocab_threshold = 5 ## hyperparam  

        train_captions = []
        for idx,cap in self.captions_dict.items():   
            train_captions.append(cap) 

        # print(train_captions)
        
        word_counts = {} 
        for cap in train_captions: 
            for word in cap.split(' '):
                word_counts[word] = word_counts.get(word, 0) + 1 

        vocab = [w for w in word_counts if word_counts[w] >= vocab_threshold] 
        # vocab = [w for w in word_counts] 

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
        # global word_map 

        word_map = dict(zip(iter(vocab), range(len(vocab))))
        img_capt_lst_tensor = []

        lookup_tensor = torch.tensor([word_map[word] if word in vocab else word_map['<unkn>'] for word in img_caption_list.split()])
        # lookup_tensor = torch.tensor([word_map[word] for word in img_caption_list.split() if word in vocab])

        # print(word_map) 
        # print(lookup_tensor)
        return lookup_tensor

    def vocab_dict(self):
        vocab = self.vocab 
        # Generate tensors
        word_map = dict(zip(iter(vocab), range(len(vocab))))
        return word_map

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


class ImageCaptionsDataset_test(Dataset):

    def __init__(self, img_dir, img_transform=None):

        self.img_dir = img_dir
        self.img_transform = img_transform 
        self.image_ids = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx] 
        image = Image.open(os.path.join(self.img_dir,img_name)) 
    
        if self.img_transform:
            image = self.img_transform(image)

        return image, img_name



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
        # for c in list(self.resnet.children())[fine_tune_last:]:
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
    def __init__(self, vocab_size, max_seq_length,  hidden_size, drop_pr=0.5, num_layers=1):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, hidden_size) 
        self.attention = attention_mod(hidden_size) 
        self.rnn = nn.GRU(hidden_size * 2, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, vocab_size)  
        self.max_seq_length = max_seq_length
        self.drop_pr = nn.Dropout(p=drop_pr, inplace = True)
        self.hidden_size = hidden_size

    def forward_step(self, input_, last_hidden, encoder_outputs):
        emb = self.emb(input_.transpose(0, 1)) 
        emb = self.drop_pr(emb)
        attn = self.attention(last_hidden, encoder_outputs) 
        context = self.drop_pr(attn.bmm(encoder_outputs).transpose(0, 1))

        # print(emb.shape) 
        # print(context.shape)
        rnn_input = torch.cat((emb, context), dim=2)

        outputs, hidden = self.rnn(rnn_input, last_hidden)

        if outputs.requires_grad:
            outputs.register_hook(lambda x: x.clamp(min=-10, max=10))

        # outputs = self.out(outputs.contiguous().squeeze(0)).log_softmax(1)
        outputs = self.out(outputs.contiguous().squeeze(0)) ## logits sending ## can use above one stop from under flow 

        return outputs, hidden

    def forward(self, x, encoder_outputs = None, train = True):  
        # print(x.shape)
        batch_size = x.shape[0]
        outputs = [] 
        self.rnn.flatten_parameters() 
        decoder_hidden = torch.zeros(1, batch_size, self.hidden_size, device=encoder_outputs.device) 
        # print(x.shape)  
        
        if train: 
            for di in range(self.max_seq_length-1):
                decoder_input = x[:, di].unsqueeze(1) 

                decoder_output, decoder_hidden = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs)

                step_output = decoder_output.squeeze(1)  
                outputs.append(step_output)

            outputs = torch.stack(outputs).permute(1, 0, 2) 

            return outputs, decoder_hidden

        else: 
            decoder_input = x[:, 0].unsqueeze(1) ## No teacher forcing 
            # print(decoder_input)
            # decoder_input = torch.tensor([[0]], device = device)
            for di in range(self.max_seq_length):

                decoder_output, decoder_hidden = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs)
                
                # print(decoder_output.shape)
                step_output = decoder_output.squeeze(1)
                outputs.append(step_output) 
                # print(step_output.shape) # torch.Size([1, 2120])
                decoder_input = step_output.topk(1)[1]
                # print(decoder_input.shape) # torch.Size([1, 1])

            outputs = torch.stack(outputs).permute(1, 0, 2) 
            return outputs, decoder_hidden

    def forward_test(self, encoder_outputs): 
        
        batch_size = 1
        outputs = [] 
        self.rnn.flatten_parameters() 
        decoder_hidden = torch.zeros(1, batch_size, self.hidden_size, device=encoder_outputs.device) 

        decoder_input = torch.tensor([[0]], device = device)
        for di in range(self.max_seq_length):

            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs)
            
            # print(decoder_output.shape)
            step_output = decoder_output.squeeze(1)
            outputs.append(step_output) 
            # print(step_output.shape) # torch.Size([1, 2120])
            decoder_input = step_output.topk(1)[1]
            # print(decoder_input.shape) # torch.Size([1, 1])

        outputs = torch.stack(outputs).permute(1, 0, 2) 
        return outputs, decoder_hidden

    # def get_bs_pred(self, x, encoder_outputs = None):
 
    #     batch_size = x.shape[0]
    #     outputs = []
    #     self.rnn.flatten_parameters()  
    #     decoder_hidden = torch.zeros(1, batch_size, self.hidden_size, device=device)   

    #     decoder_input = x[:, 0].unsqueeze(1) ## No teacher forcing 
    #     for di in range(self.max_seq_length):

    #         decoder_output, decoder_hidden = self.forward_step(
    #             decoder_input, decoder_hidden, encoder_outputs)
            
    #         # print(decoder_output.shape)
    #         step_output = decoder_output.squeeze(1)
    #         outputs.append(step_output) 
    #         # print(step_output.shape) # torch.Size([1, 2120])
    #         decoder_input = step_output.topk(1)[1]
    #         # print(decoder_input.shape) # torch.Size([1, 1])

    #     outputs = torch.stack(outputs).permute(1, 0, 2) 
    #     return outputs, decoder_hidden


class ImageCaptionsNet_mod(nn.Module):
    def __init__(self, img_width, img_height, hidden_size, vocab_size, max_len):
        super().__init__() 

        self.incept = resnet(fine_tune_last=3)

        f = self.incept(torch.rand(32, 3, img_height, img_width))  
        self._fc = f.size(1)
        self._fh = f.size(2)
        self._fw = f.size(3) 
        self.onehot_x = OneHot(self._fh)
        self.onehot_y = OneHot(self._fw) 
        self.encode_emb = nn.Linear(self._fc + self._fh + self._fw, hidden_size)
        self.decoder = decoder(vocab_size, max_len, hidden_size) 

    def forward(self, input_, target_seq=None, train = True):  
        encoder_outputs = self.incept(input_) 
        b, fc, fh, fw = encoder_outputs.size() 
        x, y = torch.meshgrid(torch.arange(fh, device=device), torch.arange(fw, device=device))
        h_loc = self.onehot_x(x)
        w_loc = self.onehot_y(y) 
        loc = torch.cat([h_loc, w_loc], dim=2).unsqueeze(0).expand(b, -1, -1, -1)
        encoder_outputs = torch.cat([encoder_outputs.permute(0, 2, 3, 1), loc], dim=3)
        encoder_outputs = encoder_outputs.contiguous().view(b, -1, self._fc + self._fh + self._fw)
        encoder_outputs = self.encode_emb(encoder_outputs)  
        # print(encoder_outputs.shape)  # torch.Size([512, 1, 2048])
        # print(target_seq.shape) # torch.Size([512, 10])
        decoder_outputs, decoder_hidden = self.decoder(target_seq, encoder_outputs=encoder_outputs, train= train)
        return decoder_outputs
    
    def enc_bs(self, input):
        encoder_outputs = self.incept(input)  
        # print(encoder_outputs.shape)
        b, fc, fh, fw = encoder_outputs.size() 
        x, y = torch.meshgrid(torch.arange(fh, device=device), torch.arange(fw, device=device)) 
        # print(x, y )
        h_loc = self.onehot_x(x)
        w_loc = self.onehot_y(y) 
        loc = torch.cat([h_loc, w_loc], dim=2).unsqueeze(0).expand(b, -1, -1, -1)
        encoder_outputs = torch.cat([encoder_outputs.permute(0, 2, 3, 1), loc], dim=3)
        encoder_outputs = encoder_outputs.contiguous().view(b, -1, self._fc + self._fh + self._fw)
        encoder_outputs = self.encode_emb(encoder_outputs)  
        # print(encoder_outputs.shape)
        return encoder_outputs


def blue_score(actual, user, sentence = True):  
    score = np.mean([nltk.translate.bleu_score.sentence_bleu([list(actual[i])], list(user[i])) for i in range(len(user))])    
    return score 

def read_raw_captions(tsv_path, val_indices):
        captions_list = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for img_caption_line in f.readlines(): 
                img_captions = img_caption_line.strip().split('\t')  
                captions_list.append(img_captions)
                
        captions_list = captions_list[:val_indices]
        return captions_list


if __name__ == '__main__': 

    img_transform = transforms.Compose([transforms.Resize(size=(256,256)), transforms.ToTensor(),  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 

    # Set the captions tsv file path
    CAPTIONS_FILE_PATH = args.train_tsv_path
    # print('preprocessing....captions')  
    captions_preprocessing_obj = CaptionsPreprocessing(CAPTIONS_FILE_PATH) 
    vocab = captions_preprocessing_obj.generate_vocabulary() 
    vocab_size = len(vocab)
    vocab_dict = captions_preprocessing_obj.vocab_dict()  
    # print(vocab_size)

    # print(max_len_cap)
    net = ImageCaptionsNet_mod(img_width=256, img_height=256, hidden_size=64, vocab_size=vocab_size, max_len=max_len_cap) ## 2 adding for end and start token
    net = net.to(device)

    IMAGE_DIR = args.image_dir_train ## train image directory  
    TEST_IMAGE_DIR = args.image_dir_test

    # Creating the Dataset
    dataset = ImageCaptionsDataset(
        IMAGE_DIR, captions_preprocessing_obj.captions_dict, img_transform=img_transform,
        captions_transform=captions_preprocessing_obj.captions_transform
    ) 

    test_dataset = ImageCaptionsDataset_test(
        TEST_IMAGE_DIR,  img_transform=img_transform) 
    
    # Define your hyperparameters
    NUMBER_OF_EPOCHS = 3000
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    VAL_BATCH_SIZE = 1 
    TEST_BATCH_SIZE = 1
    NUM_WORKERS = 0 # Parallel threads for dataloading
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)   
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.5)


    def collate_fn(data): 
        images = [data[sample]['image'] for sample in range(len(data))] 
        captions = [data[sample]['captions'] for sample in range(len(data))]  
        # print(captions)
        lengths = [len(cap) for cap in captions]   
        # print(lengths)
        # print(max(lengths))
        # Change the length of all the captions in one batch to the max of all the captions. 
        targets = torch.zeros(len(captions), max(lengths)).long() 
        # print(captions)
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end] 
        # images = torch.cat([x.float() for x in images])
        # print(images[0].shape)
        images = torch.stack([x.float() for x in images], dim=0)
        return images, targets, lengths 

    validation_split = 0.1
    dataset_size = len(dataset) 
    indices = list(range(dataset_size)) 
    split = int(np.floor(validation_split * dataset_size)) 
    shuffle_dataset = True
    if shuffle_dataset:
        np.random.seed(14)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]   
    # print(val_indices)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices) 
    # print(valid_sampler)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, 
                                            sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn) 
    val_loader = DataLoader(dataset, batch_size=VAL_BATCH_SIZE,
                                                    sampler=valid_sampler, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn) 

    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS)

    
    if args.train:

        best_val_loss = np.inf 
        val_check = 1
        for epoch in tqdm(range(NUMBER_OF_EPOCHS)): 
            train_epoch_loss = 0
            for batch_idx, sample in tqdm(enumerate(train_loader)):
                net.zero_grad() 
                net.train() 
                image_batch, captions_batch, lengths = sample
                # print(image_batch)
                # print(captions_batch)

                # image_batch, captions_batch = image_batch.to(device, dtype=torch.float), captions_batch.to(device)
                image_batch = image_batch.to(device, dtype=torch.float)
                # print(captions_batch[:,1:].shape)  
                # print(captions_batch)
                # max_len_cap = torch.as_tensor(max_len_cap, dtype=torch.int64, device='cuda')
        
                # print(image_batch.shape) 
                # print(captions_batch.shape)
                # print(lengths) 

                ## exp 
                train = torch.zeros(len(captions_batch), max(lengths)-1)
                for i, cap in enumerate(captions_batch):
                    train[i]=np.delete(cap.data, lengths[i]-1)
                
                train = train.cuda().to(dtype = torch.long)
                lengths = [l-1 for l in lengths]

                # print(image_batch.shape) 
                # print(train.shape)
 
                # output_captions = net(image_batch, captions_batch)  
                output_captions = net(image_batch, train) 

                output_captions = pack_padded_sequence(output_captions, lengths, batch_first=True,enforce_sorted = False)[0]  
                # print(output_captions.shape)
                captions_batch = captions_batch[:, 1:].cuda()
                captions_batch = pack_padded_sequence(captions_batch, lengths, batch_first=True,enforce_sorted = False)[0]  
                # print(captions_batch.shape)

                # loss = loss_function(output_captions.reshape(-1, vocab_size), captions_batch.view(-1)) 
                loss = loss_function(output_captions, captions_batch) 
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
                    for batch_idx, val_sample in tqdm(enumerate(val_loader)):
                        # val_image_batch, val_captions_batch = val_sample['image'], val_sample['captions']

                        val_image_batch, val_captions_batch, lengths = val_sample 

                        # print(lengths)

                        # print(val_image_batch.shape) 
                        # print(val_captions_batch.shape)

                        # val_image_batch, val_captions_batch = val_image_batch.to(device, dtype=torch.float), val_captions_batch.to(device)

                        val_image_batch = val_image_batch.to(device, dtype=torch.float)


                        val = torch.zeros(len(val_captions_batch), max(lengths)-1)
                        for i, cap in enumerate(val_captions_batch):
                            val[i]=np.delete(cap.data, lengths[i]-1)
                        
                        val = val.cuda().to(dtype = torch.long)
                        lengths = [l-1 for l in lengths]

                        # val_captions_batch = val_captions_batch[:, 1:].cuda()

                        val_output_captions = net(val_image_batch, val, train = False)

                        val_captions_batch = val_captions_batch[:, 1:].cuda()
                        val_captions_batch = pack_padded_sequence(val_captions_batch, lengths, batch_first=True,enforce_sorted = False)[0] 
                        val_output_captions = pack_padded_sequence(val_output_captions, lengths, batch_first=True,enforce_sorted = False)[0] 
                        
                        # val_loss = loss_function(val_output_captions.view(-1, vocab_size), val_captions_batch.view(-1)) 
                        val_loss = loss_function(val_output_captions, val_captions_batch) 
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

    else:
        # print('************')
        net.load_state_dict(torch.load(args.model_load_path)) 
        net.eval() 

        with torch.no_grad():
            if args.test:
                # pred_captions_list = []  
                with open('output.csv', 'w') as f:
                    for test_sample in tqdm(test_loader):
                        test_image_batch, img_name = test_sample 
                        # print(img_name)
                        # print(test_image_batch)
                        test_image_batch= test_image_batch.to(device, dtype=torch.float)
                        pred_caption = greedy_search(net, test_image_batch, vocab_dict) 
                        # print(pred_caption)
                        pred_caption = [st for st in pred_caption if st!= '<en>']
                        pred_caption = " ".join(pred_caption) 
                        # print(pred_caption)
                        # pred_captions_list.append(pred_caption)  
                        f.write(img_name[0] + "\t" + pred_caption + "\n")
                print('done')
                    
            elif args.val:
                pred_captions_list = [] 
                for val_sample in tqdm(val_loader):
                    val_image_batch, val_captions_batch, lengths = val_sample
                    # print(val_captions_batch)
                    val_image_batch, val_captions_batch = val_image_batch.to(device, dtype=torch.float), val_captions_batch.to(device) 
                    # pred_caption = beam_search_pred(net, val_image_batch, val_captions_batch, vocab_dict, batch_size=VAL_BATCH_SIZE) 
                    pred_caption = greedy_search(net, val_image_batch, val_captions_batch, vocab_dict, batch_size=VAL_BATCH_SIZE) 
                    # print(pred_caption) 
                    pred_captions_list.append(pred_caption) 
                actual_caption_list = read_raw_captions(args.train_tsv_path, val_indices) 
                score = blue_score(actual_caption_list, pred_captions_list) 
                print(score)
                