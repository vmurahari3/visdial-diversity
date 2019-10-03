import os
import sys
import json
import h5py
import numpy as np
from timeit import default_timer as timer

import torch
from torch.autograd import Variable

import options
import visdial.metrics as metrics
from visdial.metrics import NDCG, scores_to_ranks
from utils import utilities as utils
from dataloader import VisDialDataset
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import pairwise_distances
from six.moves import range

def rankOptions(options, gtOptions, scores):
    '''Rank a batch of examples against a list of options.'''
    # Compute score of GT options in 'scores'
    gtScores = scores.gather(1, gtOptions.unsqueeze(1))
    # Sort all predicted scores
    sortedScore, _ = torch.sort(scores, 1)
    # In sorted scores, count how many are greater than the GT score
    ranks = torch.sum(sortedScore.gt(gtScores).float(), 1)
    return ranks + 1

def rankABot(aBot, dataset, split, scoringFunction, exampleLimit=None, useNDCG=False):
    '''
        Evaluate A-Bot performance on ranking answer option when it is
        shown ground truth image features, captions and questions.

        Arguments:
            aBot    : A-Bot
            dataset : VisDialDataset instance
            split   : Dataset split, can be 'val' or 'test'

            scoringFunction : A function which computes negative log
                              likelihood of a sequence (answer) given log
                              probabilities under an RNN model. Currently
                              utils.maskedNll is the only such function used.
            exampleLimit    : Maximum number of data points to use from
                              the dataset split. If None, all data points.
    '''

    batchSize = dataset.batchSize
    numRounds = dataset.numRounds
    if exampleLimit is None:
        numExamples = dataset.numDataPoints[split]
    else:
        numExamples = exampleLimit

    numBatches = (numExamples - 1) // batchSize + 1

    original_split = dataset.split
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=1,
        collate_fn=dataset.collate_fn)

    # sparse_metrics = SparseGTMetrics()
    ndcg = None
    if useNDCG:
        ndcg = NDCG()
    ranks_json = []

    totalLoss, totalTokens = 0, 0
    ranks = []
    logProbsAll = [[] for _ in range(numRounds)]
    start_t = timer()

    getImgFileName = lambda x: dataset.data['%s_img_fnames' % split][x]
    getImgId = lambda x: int(getImgFileName(x)[:-4][-12:])

    for idx, batch in enumerate(dataloader):
        if idx == numBatches:
            break

        if dataset.useGPU:
            batch = {
                key: v.cuda() if hasattr(v, 'cuda') else v
                for key, v in batch.items()
            }
        else:
            batch = {
                key: v.contiguous() if hasattr(v, 'cuda') else v
                for key, v in batch.items()
            }

        image = Variable(batch['img_feat'], volatile=True)
        caption = Variable(batch['cap'], volatile=True)
        captionLens = Variable(batch['cap_len'], volatile=True)
        questions = Variable(batch['ques'], volatile=True)
        quesLens = Variable(batch['ques_len'], volatile=True)
        answers = Variable(batch['ans'], volatile=True)
        ansLens = Variable(batch['ans_len'], volatile=True)
        options = Variable(batch['opt'], volatile=True)
        optionLens = Variable(batch['opt_len'], volatile=True)

        gtRelevance = None
        round_id = None
        img_ids = None
        correctOptionInds = None

        if split != 'test':
            correctOptionInds = Variable(batch['ans_id'], volatile=True)

        if split == 'val' and useNDCG:
            # read in gtRelevance and round
            gtRelevance = Variable(batch['gt_relevance'],volatile=True)
            round_id = Variable(batch['round_id'],volatile=True)
            img_ids = Variable(batch['image_id'], volatile=True)

        if split == 'test':
            img_ids = [getImgId(x) for x in batch['index']]

        aBot.reset()
        aBot.observe(-1, image=image, caption=caption, captionLens=captionLens)
        log_probs_rounds = []
        for round in range(numRounds):
            aBot.observe(
                round,
                ques=questions[:, round],
                quesLens=quesLens[:, round],
                ans=answers[:, round],
                ansLens=ansLens[:, round])
            logProbs = aBot.evalOptions(options[:, round],
                                    optionLens[:, round], scoringFunction)
            if useNDCG:
                log_probs_rounds.append(logProbs.unsqueeze(1))
            logProbsCurrent = aBot.forward()
            logProbsAll[round].append(
                scoringFunction(logProbsCurrent,
                                answers[:, round].contiguous()))
            if split != 'test':
                batchRanks = rankOptions(options[:, round],
                                         correctOptionInds[:, round], logProbs)
                ranks.append(batchRanks)
        batch['num_rounds'] = batch['num_rounds'].squeeze(1)
        output = None
        if useNDCG or split == 'test':

            output = torch.cat(log_probs_rounds,dim=1)
            ranks_cur = scores_to_ranks(output)

            for i in range(len(img_ids)):
                # cast into types explicitly to ensure no errors in schema
                # round ids are 1-10, not 0-9
                # "ranks": [rank.data[0] for rank in ranks_cur[i][batch["num_rounds"][i] - 1]]

                if split == "test":
                    ranks_json.append({
                        "image_id": img_ids[i],
                        "round_id": int(batch["num_rounds"][i]),
                        "ranks": ranks_cur[i][batch["num_rounds"][i] - 1].data.cpu().tolist()
                    })
                else:
                    for j in range(numRounds):
                        ranks_json.append({
                            "image_id": img_ids[i].data[0],
                            "round_id": int(j + 1),
                            "ranks": [rank.data[0] for rank in ranks_cur[i][j]]
                        })

        if split == "val":
            # sparse_metrics.observe(output, correctOptionInds)
            if "gt_relevance" in batch and useNDCG:
                indices = torch.arange(output.shape[0]).long().cpu().numpy()
                round_id_numpy = round_id.long().cpu().data.numpy()
                round_id_numpy = round_id_numpy.reshape(-1)
                output = output.cpu().data.numpy()
                output = output[indices, round_id_numpy-1, :]
                output = Variable(torch.from_numpy(output),volatile=True)
                ndcg.observe(output, gtRelevance)

        end_t = timer()
        delta_t = " Rate: %5.2fs" % (end_t - start_t)
        start_t = end_t
        progressString = "\r[Abot] Evaluating split '%s' [%d/%d]\t" + delta_t
        sys.stdout.write(progressString % (split, idx + 1, numBatches))
        sys.stdout.flush()

    sys.stdout.write("\n")
    dataloader = None
    print("Sleeping for 3 seconds to let dataloader subprocesses exit...")
    dataset.split = original_split

    if split == 'test':
        # dump eval AI file
        dir_out = 'predictions.txt'
        json.dump(ranks_json, open(dir_out, "w"))
        return

    ranks = torch.cat(ranks, 0)
    rankMetrics = metrics.computeMetrics(ranks.cpu())

    logProbsAll = [torch.cat(lprobs, 0).mean() for lprobs in logProbsAll]
    roundwiseLogProbs = torch.cat(logProbsAll, 0).data.cpu().numpy()
    logProbsMean = roundwiseLogProbs.mean()
    rankMetrics['logProbsMean'] = logProbsMean

    if split == "val" and useNDCG:
        rankMetrics.update(ndcg.retrieve(reset=True))
        for metric_name, metric_value in rankMetrics.items():
            print(f"{metric_name}: {metric_value}")
    return rankMetrics
