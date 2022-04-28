import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import get_pad_mask, get_subsequent_mask


class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        # Store bos start character
        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]).cuda())

        # Create a tensor for [beam size, max_seq_len] note that the first column of the matrix is BOS
        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx

        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self, trg_seq, enc_output, src_mask):
        """
        trg_seq :  generate target sentence ,[beam size ,current target length, hidden size]
        if batch size!=1，[beam size * batch size,current target length, hidden size]
        stage of prediction batch size 为 1。
        sec_output : encoder output [batch size, seq length, hidden size],
        """

        # generate mask
        trg_mask = get_subsequent_mask(trg_seq)
        trg_mask = trg_mask.cuda()
        trg_seq = trg_seq.cuda()

        # tar_seq change [beam size, current target length, hidden size]
        dec_output = self.model.decoder(trg_seq, enc_output, trg_mask)

        #  hidden state -> traget vocab, 即[hidden size -> vocab size]
        return F.softmax(dec_output, dim=-1)

    def _get_init_state(self, src_seq, src_mask):


        # beam_size  represent width
        beam_size = self.beam_size

        # encoder output
        enc_output = self.model.encoder(src_seq)

        # dec_ouput is the output corresponding to the character BOS
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)

        # Get the top K characters of bos character generation probability
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)#(1,beam)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()

        # The second column of the matrix stores the largest beam size tokens
        gen_seq[:, 1] = best_k_idx[0]

        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1

        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        # 得到 [beam size, current targen length, vocab size] - > [beam size, 1, beam size]
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)#(beam,beam)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)#(beam,beam)

        # Get the best k candidates from k^2 candidates.
        # The optimal k candidates were obtained from beam size * beam size results
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)#(beam)

        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        # Find the sequence of top K and cover the original sequence in gen_seq
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, src_seq):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha


        with torch.no_grad():
            src_mask = get_pad_mask(src_seq, src_pad_idx)
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)

            ans_idx = 0  # default
            for step in range(2, max_seq_len):  # decode up to max length
                # It can be seen that the decoder input is before t time including t time.
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)

                # Get the optimal beam size sentences and corresponding score
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                # Determines if the end character EOS is encountered
                eos_locs = gen_seq == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                seq_lens.cuda()
                scores = scores.cuda()
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float().cuda() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()