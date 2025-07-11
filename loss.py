import torch
import torch.nn.functional as F
import numpy as np

from data_utils import phoneme_inventory
from utils import align_from_distances

def dtw_loss(predictions, phoneme_predictions, speech_features, phoneme_targets, silents, phoneme_loss_weight, target_lengths, est_lengths, phoneme_eval=False, phoneme_confusion=None, dimension_norm=False): 
    if phoneme_confusion is None:
        phoneme_confusion = np.zeros((len(phoneme_inventory),len(phoneme_inventory)))
    device = predictions.device
    losses = []
    losses_dist = []
    losses_ph = []
    correct_phones = 0
    total_length = 0
    for pred, y, pred_phone, y_phone, silent, target_length, est_length in zip(predictions, speech_features, phoneme_predictions, phoneme_targets, silents, target_lengths, est_lengths):
        assert len(pred.size()) == 2 and len(y.size()) == 2
        pred = pred[:est_length] if silent else pred[:target_length]
        y = y[:target_length]
        pred_phone = pred_phone[:est_length] if silent else pred_phone[:target_length]
        y_phone = y_phone[:target_length]
        D = 1 if not dimension_norm else torch.sqrt(torch.tensor([pred.shape[-1]], dtype=torch.float32)).to(device)
        y_phone = y_phone.to(device)

        if silent:
            dists = torch.cdist(pred.unsqueeze(0), y.unsqueeze(0))
            dists = dists.squeeze(0)/D

            # pred_phone (seq1_len, 48), y_phone (seq2_len)
            # phone_probs (seq1_len, seq2_len)
            pred_phone = F.log_softmax(pred_phone, -1)
            phone_lprobs = pred_phone[:,y_phone]


            costs = dists*0.5 + phoneme_loss_weight * -phone_lprobs / D

            alignment = align_from_distances(costs.T.cpu().detach().numpy())

            loss = costs[alignment,range(len(alignment))].sum()
            loss_dist = dists[alignment,range(len(alignment))].sum()
            loss_ph = -phone_lprobs[alignment,range(len(alignment))].sum()

            if phoneme_eval:
                alignment = align_from_distances(costs.T.cpu().detach().numpy())

                pred_phone = pred_phone.argmax(-1)
                correct_phones += (pred_phone[alignment] == y_phone).sum().item()

                for p, t in zip(pred_phone[alignment].tolist(), y_phone.tolist()):
                    phoneme_confusion[p, t] += 1
        else:
            assert y.size(0) == pred.size(0)

            dists = F.pairwise_distance(y, pred)/D
            loss_dist = dists.sum()

            assert len(pred_phone.size()) == 2 and len(y_phone.size()) == 1
            loss_ph = F.cross_entropy(pred_phone, y_phone, reduction='sum')
            loss = loss_dist*0.5 + phoneme_loss_weight * loss_ph / D 

            if phoneme_eval:
                pred_phone = pred_phone.argmax(-1)
                correct_phones += (pred_phone == y_phone).sum().item()

                for p, t in zip(pred_phone.tolist(), y_phone.tolist()):
                    phoneme_confusion[p, t] += 1

        losses.append(loss)
        losses_dist.append(loss_dist)
        losses_ph.append(loss_ph)
        total_length += y.size(0)
    L = sum(losses)/total_length
    L_dist = sum(losses_dist)/total_length
    L_ph = sum(losses_ph)/total_length

    return L, L_dist, L_ph, correct_phones/total_length
