# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 18:16:00 2021

@author: nsahu
"""
import torch
import torch.nn.functional as F

def beam_search_pred(model, image,  vocab_dict, beam_width=3, set_len=80):
    end = vocab_dict['<en>']
    enc_feat = model.enc_bs(image) 
    # print(enc_feat.shape) # torch.Size([1, 512])
    enc_feat = enc_feat.unsqueeze(1)
    
    #Generating the first token
    out, hidden = model.get_bs_pred(enc_feat)
    out = F.log_softmax(out, dim=1)
    # print(out.shape) # torch.Size([1, 1837])

    log_pr, indexes =  torch.topk(out, beam_width)
    
    log_pr = log_pr.cpu().detach().numpy().squeeze(0)
    indexes = indexes.cpu().detach().numpy()
    # indexes = indexes.cpu().detach().numpy().squeeze(0)
    
    tmp = []
    result = []
    # for ind in range(len(indexes)):
    for ind in range(indexes.shape[1]):
        # tmp.append([[indexes[ind]], log_pr[ind], hidden])
        tmp.append([[torch.tensor(indexes[:,ind])], log_pr[ind], hidden])
    # print(hidden[0].shape)
        
    #Starting the loop
    new = []
    while(len(result) < set_len):
        for k in tmp:
            # print('***********')
            out, hidden = model.get_bs_pred(k[0][-1].cuda(), k[-1])
            # out, hidden = model.get_bs_pred(torch.tensor(k[0][-1]).cuda(), k[-1])
            out = F.log_softmax(out, dim=1) 
            # print(out.shape)
            # print('>>>>>')
            log_pr, indexes =  torch.topk(out, beam_width)
            
            log_pr = log_pr.cpu().detach().numpy().squeeze(0)
            indexes = indexes.cpu().detach().numpy() 

            # print(indexes) 
            # for ind in range(len(indexes)):
            for ind in range(indexes.shape[1]):
                # new.append([[k[0], [indexes[ind]]], (log_pr[ind] + k[1]), hidden])
                new.append([[k[0], [torch.tensor(indexes[:,ind])]], log_pr[ind] + k[1], hidden]) ## hidden is same or what?? 
            
        for i in new:
            i[0] = [wi for sl in i[0] for wi in sl]
            tmp.append(i) 
            
        new = []
        tmp = tmp[beam_width:]
        tmp = sorted(tmp, reverse=True, key=lambda l: l[1])
        tmp = tmp[:beam_width]
        
        result = tmp[0][0]
        #check if end token is generated
        if (result[-1] == end):
            break
    
    ind_to_word = {}
    for k,v in vocab_dict.items():
        ind_to_word[v] = k
    
    result_caption = []
    result = result[1:]
    for r in result:
        result_caption.append(ind_to_word[r.item()])

    print(result_caption) 
    return result_caption
    
def greedy_search(model, image, vocab_dict, set_len = 10):
    enc_feat = model.enc_bs(image) 
    # print(enc_feat.shape) # torch.Size([1, 512]) ## N,L,Hin >> 1,1,512
    enc_feat = enc_feat.unsqueeze(1) ## addon 
    
    #Generating the first token
    out, hidden = model.get_bs_pred(enc_feat)
    out_pr = F.log_softmax(out, dim=1) 

    # print(out_pr.shape) # torch.Size([1, 1837])

    log_pr, indx = torch.topk(out_pr, 1)   
    # print(indx.shape) # torch.Size([1, 1]) 
    indx = indx.cpu().detach().numpy().squeeze(0)  
    # print(indx) 

    ind_to_word = {}
    for k,v in vocab_dict.items():
        ind_to_word[v] = k 

    result = []
    result.append(indx)  
    while(len(result) < set_len): 
        # print(torch.tensor(indx).cuda().shape) # torch.Size([1, 1])
        out, hidden =  model.get_bs_pred(torch.tensor(indx).cuda(), hidden)  
        out_pr = F.log_softmax(out, dim=1) 
        log_pr, indx = torch.topk(out_pr, 1)  
        indx = indx.cpu().detach().numpy().squeeze(0)
        if (ind_to_word[indx.item()]) == '<en>':
            break
        result.append(indx)  

    result_caption = []
    for r in result:
        # print(r)
        result_caption.append(ind_to_word[r.item()]) 
    
    # print(result_caption) 
    return result_caption

    
    
