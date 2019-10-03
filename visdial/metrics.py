import torch
import numpy as np
# static list of metrics
metricList = ['r1', 'r5', 'r10', 'mean', 'mrr']
# +1 - greater the better
# -1 - lower the better
trends = [1, 1, 1, -1, -1, 1]

def evaluateMetric(ranks, metric):
    ranks = ranks.data.numpy()
    if metric == 'r1':
        ranks = ranks.reshape(-1)
        return 100 * (ranks == 1).sum() / float(ranks.shape[0])
    if metric == 'r5':
        ranks = ranks.reshape(-1)
        return 100 * (ranks <= 5).sum() / float(ranks.shape[0])
    if metric == 'r10':
        # ranks = ranks.view(-1)
        ranks = ranks.reshape(-1)
        # return 100*torch.sum(ranks <= 10).data[0]/float(ranks.size(0))
        return 100 * (ranks <= 10).sum() / float(ranks.shape[0])
    if metric == 'mean':
        # ranks = ranks.view(-1).float()
        ranks = ranks.reshape(-1).astype(float)
        return ranks.mean()
    if metric == 'mrr':
        # ranks = ranks.view(-1).float()
        ranks = ranks.reshape(-1).astype(float)
        # return torch.reciprocal(ranks).mean().data[0]
        return (1 / ranks).mean()

def computeMetrics(ranks):
    results = {metric: evaluateMetric(ranks, metric) for metric in metricList}
    return results

def scores_to_ranks(scores: torch.Tensor):
    """Convert model output scores into ranks."""
    batch_size, num_rounds, num_options = scores.size()
    scores = scores.view(-1, num_options)

    # sort in descending order - largest score gets highest rank
    sorted_ranks, ranked_idx = scores.sort(1, descending=True)

    # i-th position in ranked_idx specifies which score shall take this position
    # but we want i-th position to have rank of score at that position, do this conversion
    ranks = ranked_idx.clone().fill_(0)

    for i in range(ranked_idx.size(0)):
        for j in range(num_options):
            ranks[i,ranked_idx[i][j].data[0]] = j
    # convert from 0-99 ranks to 1-100 ranks
    ranks += 1
    ranks = ranks.view(batch_size, num_rounds, num_options)
    return ranks

class NDCG(object):
    def __init__(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0
        self.ndcg_vals = []

    def observe(self,
                predicted_scores: torch.Tensor,
                target_relevance: torch.Tensor):
        """
        Observe model output scores and target ground truth relevance and accumulate NDCG metric.

        Parameters
        ----------
        predicted_scores: torch.Tensor
            A tensor of shape (batch_size, num_options), because dense annotations are
            available for only one randomly picked round out of ten.
        target_relevance: torch.Tensor
            A tensor of shape same as predicted scores, indicating ground truth relevance of
            each answer option for a particular round.
        """
        predicted_scores = predicted_scores.detach()

        # shape: (batch_size, 1, num_options)
        predicted_scores = predicted_scores.unsqueeze(1)
        predicted_ranks = scores_to_ranks(predicted_scores)

        # shape: (batch_size, num_options)
        predicted_ranks = predicted_ranks.squeeze()
        batch_size, num_options = predicted_ranks.size()

        k = torch.sum(target_relevance != 0, dim=-1)

        # shape: (batch_size, num_options)
        _, rankings = torch.sort(predicted_ranks, dim=-1)
        # Sort relevance in descending order so highest relevance gets top rank.
        _, best_rankings = torch.sort(target_relevance, dim=-1, descending=True)

        # shape: (batch_size, )
        batch_ndcg = []
        for batch_index in range(batch_size):
            num_relevant = int(k.data[batch_index])
            dcg = self._dcg(
                rankings[batch_index][:num_relevant], target_relevance[batch_index]
            )
            best_dcg = self._dcg(
                best_rankings[batch_index][:num_relevant], target_relevance[batch_index]
            )
            batch_ndcg.append(dcg / best_dcg)
            self.ndcg_vals.append(dcg.data[0]/best_dcg.data[0])

        self._ndcg_denominator += batch_size
        self._ndcg_numerator += sum(batch_ndcg)

    def _dcg(self, rankings: torch.Tensor, relevance: torch.Tensor):
        rankings = rankings.cuda()
        sorted_relevance = relevance[rankings].cpu().float()
        discounts = np.log2(torch.arange(len(rankings)).float().numpy() + 2)
        discounts = torch.autograd.Variable(torch.from_numpy(discounts))
        return torch.sum(sorted_relevance / discounts, dim=-1)

    def retrieve(self, reset: bool = True):
        if self._ndcg_denominator > 0:
            metrics = {
                "ndcg": float(self._ndcg_numerator / self._ndcg_denominator),
                "ndcg_std": np.std(np.array(self.ndcg_vals))
            }
        else:
            metrics = {}

        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0
        self.ndcg_vals = []
