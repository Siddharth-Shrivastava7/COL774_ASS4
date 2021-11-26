# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 18:16:00 2021

@author: nsahu
"""
import torch
import torch.nn.functional as F

def beam_search_pred(model, image,  vocab_dict, beam_width=3, set_len=80):
    end = vocab_dict[' <en>']
    enc_feat = model.enc_bs(image)
    
    #Generating the first token
    out, hidden = model.get_bs_pred(enc_feat)
    out = F.log_softmax(out, dim=1)
    
    log_pr, indexes =  torch.topk(out, beam_width)
    
    log_pr = log_pr.cpu().detach().numpy().squeeze(0)
    indexes.pr = indexes.cpu().detach().numpy().squeeze(0)
    
    tmp = []
    result = []
    for ind in range(len(indexes)):
        tmp.append([[indexes[ind]], log_pr[ind], hidden])
        
    #Starting the loop
    new = []
    while(len(result) < set_len):
        for k in tmp:
            out, hidden = model.get_bs_pred(torch.tensor(k[0][-1]))
            out = F.log_softmax(out, dim=1)
            
            log_pr, indexes =  torch.topk(out, beam_width)
            
            log_pr = log_pr.cpu().detach().numpy().squeeze(0)
            indexes.pr = indexes.cpu().detach().numpy().squeeze(0)
            for ind in range(len(indexes)):
                new.append([k[0], [indexes[ind]], log_pr[ind] + k[1], hidden])
            
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
        result_caption.append(ind_to_word[r])
    
    return result_caption
    
