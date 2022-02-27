import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from transformer import make_std_mask, subsequent_mask
#from beamsearch import Translator
import torch.distributed as dist
from batch_beamsearch import Translator
import deepsmiles
import numpy as np
import pandas as pd
from pycocoevalcap.rouge.rouge import Rouge
error_file_ids = "./_test_error.txt"
hypethese_file ="./_test_hypethese.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
# print("当前GPU设备索引为：",torch.cuda.current_device() ) # 返回当前设备索引
#checkpoint = torch.load(checkpoint,map_location=torch.device('cpu') )

# Load word map (word2ix)

word_map = torch.load('../../Data/500wan/500wan_shuffle_DeepSMILES_word_map')
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def acc(pred_data,test_data):
    error_ids = []
    sum_count = len(pred_data)
    count = 0
    notin_count = 0
    for i in range(sum_count):
        if pred_data[i] == test_data[i][0]:
                count+=1

        else:
            error_ids.append(i)
            
            notin_count+=1
    return count/sum_count*1.0,error_ids

def save_to_file(filename,references, data,word_dic,truth=False):
    converter = deepsmiles.Converter(rings=True, branches=True)
    ture_f = open('ture.txt','w', encoding='utf-8')
    pre_f = open('pre.txt','w',encoding='utf-8')
    # pre_f.write('Smiles'+'\n')
    # ture_f.write('Smiles'+'\n')
    all_count = len(data)
    count = 0
    for i in range(all_count):#预测 全部smiles的序号
        true_ids = references[i][0]
        ids = data[i]
        if truth:
            ids = [id for id in ids if word_dic[id]!='<start>' and word_dic[id]!='<pad>' and word_dic[id]!='<end>']
        smiles = [rev_word_map[id] for id in ids if id !=word_map['<end>']]
        true_smiles = [rev_word_map[id]  for id in true_ids if id !=word_map['<end>']]
        smiles = ''.join(smiles)
        true_smiles = ''.join(true_smiles)
        try:
            decoded = converter.decode(smiles)
            count +=1
            pre_f.write(decoded+'\n')
        except:
            print('error')
            continue
        try:
            decoded = converter.decode(true_smiles)
            ture_f.write(decoded + '\n')
        except:
            print('error')
    ture_f.close()
    pre_f.close()
    return count/all_count*1.0



def greedy_decode_1(decoder, memory, max_len, start_symbol, end_symbol):
    batch_size = memory.shape[0]
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    logits = None
    for i in range(max_len - 1):

        logit, _ = decoder(memory, Variable(ys).to(device),
                           Variable(subsequent_mask(ys.size(1)).type(torch.long)).to(device))
        prob = nn.Softmax(dim=-1)(logit[:,-1])
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys,
                        next_word.type(torch.long).to(device)], dim=1)
        if i == max_len - 2:
            logits = logit
        # print('time:'+str(end))
        # if next_word == end_symbol:
        #     logits = logit
        #     break
    return ys, logits

def Greedy_decode(decoder, memory, max_len, start_symbol, end_symbol):
    batch_size = memory.shape[0]
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    logits = None
    for i in range(max_len - 1):

        logit = decoder(Variable(ys).to(device), memory,
                           Variable(subsequent_mask(ys.size(1)).type(torch.long)).to(device))
        prob = nn.Softmax(dim=-1)(logit[:,-1])
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys,
                        next_word.type(torch.long).to(device)], dim=1)
        if i == max_len - 2:
            logits = logit
        # print('time:'+str(end))
        # if next_word == end_symbol:
        #     logits = logit
        #     break
    return ys, logits
def beam_search (batch_size,Decoder, beam_size, max_seq_len,src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):

    #translator = Translator(Decoder, beam_size, max_seq_len,src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx)
    translator = Translator(batch_size, vocab_size,Decoder, beam_size, max_seq_len, src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx)

    return translator


def swin_evaluate(beam_size, encoder, decoder,test_dir):
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    encoder = encoder.to(device)
    encoder.eval()
    decoder = decoder.to(device)
    decoder.eval()
    # DataLoader
    data_transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        # transforms.RandomRotation(15),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        # transforms.RandomErasing(p=0.3,scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset_test = CaptionDataset_500wan_test(test_dir, transform=data_transform)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_test = torch.utils.data.DistributedSampler(
        dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    loader = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=512,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    num_smiles = 0
    right_smiles = 0
    beam_decoder = beam_search(512, decoder, beam_size, max_seq_len=102, src_pad_idx=0, trg_pad_idx=0,
                         trg_bos_idx=word_map['<start>'], trg_eos_idx=word_map['<end>'])

    with torch.no_grad():
        for i, (imgs, caps, caplens) in enumerate(
                tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caps_np = caps.cpu().tolist()
            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            if beam_size!=1:
                logits = beam_decoder.translate_sentence(imgs)
                logits = logits[:, 1:]
                for r in range(logits.shape[0]):
                    pre_list = []
                    for j in logits[r].tolist():
                        if j != word_map['<end>']:
                            pre_list.append(j)
                        else:
                            pre_list.append(j)
                            break
                    hypotheses.append(pre_list)


            else:
                y_preds, logits = Greedy_decode(decoder, imgs, max_len=102,
                                                start_symbol=word_map['<start>'], end_symbol=word_map['<end>'])



                _, preds = torch.max(logits, dim=-1)
                for r in range(preds.shape[0]):
                    pre_list = []
                    for j in preds[r].tolist():
                        if j != word_map['<end>']:
                            pre_list.append(j)
                        else:
                            pre_list.append(j)
                            break
                    hypotheses.append(pre_list)
            #References

            img_captions = list(
                map(lambda c: [[w for w in c if w not in {word_map['<start>'], word_map['<pad>']}]],
                    caps_np))  # remove <start> and pads
            references.extend(img_captions)


            acc_score, _ = acc(hypotheses, references)
            print(f"当前已预测{(i + 1) * 512}个，当前正确率为：{acc_score}")
        assert len(references) == len(hypotheses)

    valid = save_to_file(hypethese_file, references, hypotheses, rev_word_map)

    # Calculate BLEU-4 scores
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0, 1, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0, 0, 1, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0, 0, 0, 1))
    bleu_avg = corpus_bleu(references, hypotheses)
    acc_score, error_ids = acc(hypotheses, references)
    res = {}
    gts = {}
    for i in range(len(hypotheses)):
        res[i] = [' '.join(list(map(str,hypotheses[i])))]
        gts[i] = [' '.join(list(map(str,references[i][0])))]
    scorer = Rouge()
    rouge_score, scores = scorer.compute_score(gts, res)

    with open(error_file_ids, 'w', encoding='utf-8') as error_ids_f:  # =============
        error_ids_f.write(' '.join([str(id + 1) for id in error_ids]))
    return bleu1, bleu2, bleu3, bleu4, bleu_avg, acc_score, valid, rouge_score


def Swin_Evaluate(model_name,beam,encoder,decoder,test_dir):
    log_file = open(f"./test_result_beam_{beam}.log","a")
    bleu1,bleu2,bleu3,bleu4,bleu_avg, acc_score_1, valid, rouge_score = swin_evaluate(beam, encoder=encoder, decoder=decoder,test_dir=test_dir)

    print("\nbefore_transformer_result_bs_16_embedding_16 N_6 H_8 BLEU score @ beam size %d is %.4f %.4f %.4f %.4f. avg: %.4f. acc %.4f. valid: %.4f rouge_score:%.4f" %
        (1, bleu1, bleu2, bleu3, bleu4,bleu_avg, acc_score_1, valid, rouge_score))
    log_file.write("\n%s BLEU score @ beam size %d is %.4f %.4f %.4f %.4f. avg: %.4f. acc %.4f. Valid: %.4f rouge_score:%.4f" % (model_name, beam, bleu1, bleu2, bleu3, bleu4,bleu_avg, acc_score_1, valid, rouge_score))
if __name__ == '__main__':
    log_file = open("./test_result.log","a")
    bleu1,bleu2,bleu3,bleu4,bleu_avg, acc_score_1 = evaluate(1)

    print("\nbefore_transformer_result_bs_16_embedding_16 N_6 H_8 BLEU score @ beam size %d is %.4f %.4f %.4f %.4f. avg: %.4f. acc %.4f" % 
        (1, bleu1, bleu2, bleu3, bleu4,bleu_avg, acc_score_1))
    log_file.write("\nbefore_transformer_result_bs_16_embedding_16 N_6 H_8 BLEU score @ beam size %d is %.4f %.4f %.4f %.4f. avg: %.4f. acc %.4f" % (1, bleu1, bleu2, bleu3, bleu4,bleu_avg, acc_score_1))


