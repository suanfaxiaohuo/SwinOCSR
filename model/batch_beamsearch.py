import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import get_pad_mask, get_subsequent_mask


class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, batch_size, vocab_size, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):

        super(Translator, self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()


        self.register_buffer('init_seq', torch.ones(self.batch_size, 1).fill_(trg_bos_idx).type(torch.long).cuda())


        self.register_buffer(
            'blank_seqs',
            torch.full((self.batch_size, beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, :, 0] = self.trg_bos_idx

        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self, trg_seq, enc_output, src_mask):
        """
        trg_seq :  generate target sentenct ,[beam size ,current target length, hidden size]
        batch size!=1，[beam size * batch size,current target length, hidden size]
        stage of prediction batch size == 1。
        sec_output : encoder [batch size, seq length, hidden size],
        """
        # generate sentence mask
        trg_seq = trg_seq.reshape(self.batch_size * self.beam_size, -1)

        trg_mask = get_subsequent_mask(trg_seq)
        trg_mask = trg_mask.cuda()
        trg_seq = trg_seq.cuda()

        enc_output = enc_output.reshape(self.batch_size*self.beam_size, -1, 256)

        # tar_seq change [beam size, current target length, hidden size]

        dec_output = self.model.decoder(trg_seq, enc_output, trg_mask)

        dec_output = dec_output.view(self.batch_size, self.beam_size, -1, self.vocab_size)

        #  hidden state -> traget vocab, [hidden size -> vocab size]
        return F.softmax(dec_output, dim=-1)

    def _model_decode_init(self, trg_seq, enc_output, src_mask):
        """
        trg_seq :  generate target sentenct ,[beam size ,current target length, hidden size]
        if batch size!=1，[beam size * batch size,current target length, hidden size]
        stage of prediction batch size=1。
        sec_output : encoder [batch size, seq length, hidden size],

        """

        # generate sentence mask
        trg_mask = get_subsequent_mask(trg_seq)
        trg_mask = trg_mask.cuda()
        trg_seq = trg_seq.cuda()
        # tar_seq change  [beam size, current target length, hidden size]
        dec_output = self.model.decoder(trg_seq, enc_output, trg_mask)
        #  hidden state -> traget vocab, 即[hidden size -> vocab size]
        return F.softmax(dec_output, dim=-1)


    def _get_init_state(self, src_seq, src_mask):


        # Beam_size represent width
        beam_size = self.beam_size

        # Encoder output
        enc_output = self.model.encoder(src_seq)

        # Dec_ouput represent bos output
        dec_output = self._model_decode_init(self.init_seq, enc_output, src_mask)

        # Take bos top k char
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)#(batch,beam)

        scores = torch.log(best_k_probs).view(self.batch_size, beam_size)
        gen_seq = self.blank_seqs.clone().detach()

        # The second column of the matrix stores the largest beam size tokens
        gen_seq[:, :, 1] = best_k_idx

        enc_output = enc_output.unsqueeze(1).repeat(1, beam_size, 1, 1)
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        #assert len(scores.size()) == 1

        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        # Get [beam size, current targen length, vocab size] - > [beam size, 1, beam size]

        best_k2_probs, best_k2_idx = dec_output[:, :, -1, :].topk(beam_size)#B,b，b

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(self.batch_size, beam_size, -1) + scores.view(self.batch_size, beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        # The optimal k candidates were obtained from beam size * beam size results
        scores, best_k_idx_in_k2 = scores.view(self.batch_size, -1).topk(beam_size) #B,b

        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = None
        for i in range(len(best_k_c_idxs)):
            if best_k_idx is None:
                best_k_idx_tmp = best_k2_idx[i][best_k_r_idxs[i], best_k_c_idxs[i]]
                best_k_idx = best_k_idx_tmp.unsqueeze(0)
            else:
                best_k_idx_tmp = best_k2_idx[i][best_k_r_idxs[i], best_k_c_idxs[i]].unsqueeze(0)
                best_k_idx = torch.cat((best_k_idx, best_k_idx_tmp), 0)

        # Copy the corresponding previous tokens.
        # Find the sequence of top K and cover the original sequence in Gen SEq
        #print(best_k_r_idxs)
        #print(best_k_idx)
        #print(best_k_idx.shape)
        for i in range(len(gen_seq)):
            gen_seq[i, :, :step] = gen_seq[i, best_k_r_idxs[i], :step]
        #gen_seq[:, :, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, :, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, src_seq):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha
        src_seq = self.model.encoder_dim(src_seq)
        with torch.no_grad():
            src_mask = get_pad_mask(src_seq, src_pad_idx)
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)
            #print(gen_seq)
            ans_idx = 0  # default
            for step in range(2, max_seq_len):  # decode up to max length
                # It can be seen that the decoder input is before t time including t time.
                dec_output = self._model_decode(gen_seq[:, :, :step], enc_output, src_mask)
                #print(dec_output)
                # Get the optimal beam size sentences and corresponding score
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)
                #print(gen_seq)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                # Determines if the end character EOS is encountered
                eos_locs = gen_seq == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(2)
                seq_lens.cuda()
                scores = scores.cuda()
                # -- check if all beams contain eos
                if ((eos_locs.sum(2) > 0).sum(1) == beam_size).sum(0).item() == self.batch_size:
                    # TODO: Try different terminate conditions.

                    #_, ans_idx = scores.div(seq_lens.float().cuda() ** alpha).max(1)
                    _, ans_idx = scores.div(seq_lens.float().cuda()).max(1)
                    ans_idx = ans_idx.tolist()
                    break
                #print(ans_idx)
        # return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()
        # return gen_seq[range(self.batch_size),ans_idx,:seq_lens[range(self.batch_size),ans_idx]]
        return gen_seq[range(self.batch_size), ans_idx]