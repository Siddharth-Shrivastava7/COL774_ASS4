# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 18:16:00 2021

@author: nsahu
"""
import torch
import torch.nn.functional as F

def beam_search_pred(model, image, caption, vocab_dict, batch_size, hidden_size = 2048, beam_width=3, set_len=10):
    end = vocab_dict['<en>']
    encoder_outputs = model.enc_bs(image) 
    decoder_input = caption[:, 0].unsqueeze(1)

    decoder_hidden = torch.zeros(1, batch_size, hidden_size, device=encoder_outputs.device)

    #Generating the first token 
    # out, hidden = model.decoder.get_bs_pred(caption, encoder_outputs)   
    # out = F.log_softmax(out, dim=1) 

    out, hidden = model.decoder.forward_step(decoder_input, decoder_hidden, encoder_outputs)   
    out = F.log_softmax(out, dim=1)
    # print(out.shape) 
    # print('**********')    
    
    log_pr, indexes =  torch.topk(out, beam_width)   
    log_pr = log_pr.cpu().detach().numpy().squeeze(0) 
    indexes = indexes.cpu().detach().numpy() 
    # print(torch.tensor(indexes)[:,1].unsqueeze(1).shape) # torch.Size([1, 3])
    # ind = out.topk(3)[1] # torch.Size([1, 3])
    # print(ind.shape) 
    # print(indexes.shape[1]) 
    # print(torch)

    tmp = []
    result = []
    # for ind in range(len(indexes)):
    for ind in range(indexes.shape[1]):
        tmp.append([[torch.tensor(indexes[:,ind]).unsqueeze(1)], log_pr[ind], hidden]) 

    #Starting the loop
    new = [] 
    while(len(result) < set_len):
        for k in tmp:   
            # print(k[0].shape) 
            # print(torch.tensor(k[0][-1]).shape)  
            # print('>>>>>>')

            # out, hidden = model.decoder.forward_step(torch.tensor(k[0][-1]).cuda(), k[-1], encoder_outputs) 
            out, hidden = model.decoder.forward_step(k[0][-1].cuda(), k[-1], encoder_outputs) 
            out = F.log_softmax(out, dim=1) 
            
            log_pr, indexes =  torch.topk(out, beam_width)
            
            log_pr = log_pr.cpu().detach().numpy().squeeze(0)
            indexes = indexes.cpu().detach().numpy() 

            for ind in range(len(indexes)):
                # new.append([k[0], [indexes[ind]], log_pr[ind] + k[1], hidden])
                new.append([[k[0], [torch.tensor(indexes[:,ind]).unsqueeze(1)]], log_pr[ind] + k[1], hidden])
        # print('********') 
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
    # print(len(result)) 
    for r in result:
        # print(r.item())
        result_caption.append(ind_to_word[r.item()])
    # print(result_caption)
    return result_caption


def greedy_search(model, image, vocab_dict, set_len=10):
    encoder_outputs = model.enc_bs(image)  
    # print(encoder_outputs.shape) # torch.Size([1, 1, 64]) 
    output, hidden = model.decoder.forward_test(encoder_outputs)
    out = F.log_softmax(output, dim=1)  
    # print(out.shape)
    log_pr, indexes =  torch.topk(out, 1) 
    # print(indexes.shape)


    # decoder_input = caption[:, 0].unsqueeze(1)
    # # print(decoder_input.shape) # torch.Size([1, 1])
    # decoder_hidden = torch.zeros(1, batch_size, hidden_size, device=encoder_outputs.device)  
    # # print(decoder_hidden.shape) # torch.Size([1, 1, 64])

    
    # #Generating the first token
    # out, hidden = model.decoder.forward_step(decoder_input, decoder_hidden, encoder_outputs)   
    # out = F.log_softmax(out, dim=1) 
    # # print(out.shape) # torch.Size([1, 2120])

    # # indx = torch.argmax(out, dim = 1) 
    # _, indx = torch.topk(out, 1) 
    # if indx == 0: 
    #     _, indx = torch.topk(out, 6) 
    #     # print(indx)
    #     indx = indx[0][5].unsqueeze(0).unsqueeze(1)
    #     # print(indx.shape)
    # # print(indx.shape) # torch.Size([1, 1]) 
    # indx = indx.cpu().detach().numpy() 
    # # print(indx) 

    ind_to_word = {}
    for k,v in vocab_dict.items():
        ind_to_word[v] = k 

    result = []
    # result.append(indx)  
    # while(len(result) < set_len): 
    #     # print(torch.tensor(indx).cuda().shape) # torch.Size([1, 1]) 

    #     out, hidden =   model.decoder.forward_step(torch.tensor(indx).cuda(), hidden, encoder_outputs)  
    #     out = F.log_softmax(out, dim=1) 
    #     _, indx = torch.topk(out, 1)
    #     indx = indx.cpu().detach().numpy()
    #     if (ind_to_word[indx.item()]) == '<en>':
    #         break
    #     result.append(indx)  

    result_caption = []
    # print(result)
    for r in range(indexes.shape[1]):
        # print()
        # print(r)
        result_caption.append(ind_to_word[indexes[0,r].item()]) 
    
    # print(result_caption) 
    return result_caption