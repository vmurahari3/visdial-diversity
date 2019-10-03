import os
import gc
import random
import pprint
from six.moves import range
from time import gmtime, strftime
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import options
from dataloader import VisDialDataset
from torch.utils.data import DataLoader
from eval_utils.rank_answerer import rankABot
from eval_utils.rank_questioner import rankQBot
from eval_utils.rank_questioner import rankQABots
from eval_utils.dialog_generate import run_dialog
from utils import utilities as utils
from utils.visualize import VisdomVisualize

# ---------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------

# Read the command line options
params = options.readCommandLine()

# Seed rng for reproducibility
random.seed(params['randomSeed'])
torch.manual_seed(params['randomSeed'])
if params['useGPU']:
    torch.cuda.manual_seed_all(params['randomSeed'])

# Setup dataloader
splits = ['train', 'val', 'test']

dataset = VisDialDataset(params, splits)

# Params to transfer from dataset
transfer = ['vocabSize', 'numOptions', 'numRounds']
for key in transfer:
    if hasattr(dataset, key):
        params[key] = getattr(dataset, key)

# Create save path and checkpoints folder
os.makedirs('checkpoints', exist_ok=True)
os.mkdir(params['savePath'])

# Loading Modules
parameters = []
aBot = None
qBot = None

# Loading Q-Bot
if params['trainMode'] in ['sl-qbot', 'rl-full-QAf']:
    qBot, loadedParams, optim_state = utils.loadModel(params, 'qbot')
    for key in loadedParams:
        params[key] = loadedParams[key]

    if params['trainMode'] == 'rl-full-QAf' and params['freezeQFeatNet']:
        qBot.freezeFeatNet()
    # Filtering parameters which require a gradient update
    parameters.extend(filter(lambda p: p.requires_grad, qBot.parameters()))

    # parameters.extend(qBot.parameters())
# Loading A-Bot
if params['trainMode'] in ['sl-abot', 'rl-full-QAf']:
    aBot, loadedParams, optim_state = utils.loadModel(params, 'abot')
    for key in loadedParams:
        params[key] = loadedParams[key]
    parameters.extend(aBot.parameters())

# Setup pytorch dataloader
dataset.split = 'train'
dataloader = DataLoader(
    dataset,
    batch_size=params['batchSize'],
    shuffle=False,
    num_workers=params['numWorkers'],
    drop_last=True,
    collate_fn=dataset.collate_fn,
    pin_memory=False)

# Initializing visdom environment for plotting data
viz = VisdomVisualize(
    enable=bool(params['enableVisdom']),
    env_name=params['visdomEnv'],
    server=params['visdomServer'],
    port=params['visdomServerPort'])
pprint.pprint(params)
viz.addText(pprint.pformat(params, indent=4))

# Setup optimizer
if params['continue']:
    # Continuing from a loaded checkpoint restores the following
    startIterID = params['ckpt_iterid'] + 1  # Iteration ID
    lRate = params['ckpt_lRate']  # Learning rate
    print("Continuing training from iterId[%d]" % startIterID)
else:
    # Beginning training normally, without any checkpoint
    lRate = params['learningRate']
    startIterID = 0

optimizer = optim.Adam(parameters, lr=lRate)
if params['continue']:  # Restoring optimizer state
    print("Restoring optimizer state dict from checkpoint")
    optimizer.load_state_dict(optim_state)
runningLoss = None

mse_criterion = nn.MSELoss(reduce=False)

numIterPerEpoch = dataset.numDataPoints['train'] // params['batchSize']
print('\n%d iter per epoch.' % numIterPerEpoch)

if params['useCurriculum']:
    if params['continue']:
        rlRound = max(0, 9 - (startIterID // numIterPerEpoch))
    else:
        rlRound = params['numRounds'] - 1
else:
    rlRound = 0

# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------

def batch_iter(dataloader):
    for epochId in range(params['numEpochs']):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch

start_t = timer()

outNet = nn.Linear(params["rnnHiddenSize"], params["vocabSize"])
logSoftmax = nn.LogSoftmax(dim=1)

if params["useGPU"]:
    outNet.cuda()
    logSoftmax.cuda()

baseline = 0

for epochId, idx, batch in batch_iter(dataloader):
    # Keeping track of iterId and epoch
    iterId = startIterID + idx + (epochId * numIterPerEpoch)
    epoch = iterId // numIterPerEpoch
    gc.collect()
    # Moving current batch to GPU, if available
    if dataset.useGPU:
        batch = {key: v.cuda() if hasattr(v, 'cuda') \
            else v for key, v in batch.items()}

    image = Variable(batch['img_feat'], requires_grad=False)
    caption = Variable(batch['cap'], requires_grad=False)
    captionLens = Variable(batch['cap_len'], requires_grad=False)
    gtQuestions = Variable(batch['ques'], requires_grad=False)
    gtQuesLens = Variable(batch['ques_len'], requires_grad=False)
    gtAnswers = Variable(batch['ans'], requires_grad=False)
    gtAnsLens = Variable(batch['ans_len'], requires_grad=False)
    options = Variable(batch['opt'], requires_grad=False)
    optionLens = Variable(batch['opt_len'], requires_grad=False)
    gtAnsId = Variable(batch['ans_id'], requires_grad=False)

    # Initializing optimizer and losses
    optimizer.zero_grad()
    loss = 0
    qBotLoss = 0
    aBotLoss = 0
    cos_similarity_loss = 0
    huber_loss = 0
    rlLoss = 0
    featLoss = 0
    qBotRLLoss = 0
    aBotRLLoss = 0
    predFeatures = None
    initialGuess = None
    numRounds = params['numRounds']
    # numRounds = 1 # Override for debugging lesser rounds of dialog

    # Setting training modes for both bots and observing captions, images where needed
    if aBot:
        aBot.train(), aBot.reset()
        aBot.observe(-1, image=image, caption=caption, captionLens=captionLens)
    if qBot:
        qBot.train(), qBot.reset()
        qBot.observe(-1, caption=caption, captionLens=captionLens)

    # Q-Bot image feature regression ('guessing') only occurs if Q-Bot is present
    if params['trainMode'] in ['sl-qbot', 'rl-full-QAf']:
        initialGuess = qBot.predictImage()
        prevFeatDist = mse_criterion(initialGuess, image)
        featLoss += torch.mean(prevFeatDist)
        prevFeatDist = torch.mean(prevFeatDist,1)

    cum_reward = torch.zeros(params['batchSize'])
    if params['useGPU']:
        cum_reward = cum_reward.cuda()
    past_dialog_hidden = None
    cur_dialog_hidden = None

    mean_reward_batch = 0
    # calculate the mean reward value for this batch. This will be used to update baseline.
    for round in range(numRounds):
        '''
        Loop over rounds of dialog. Currently three modes of training are
        supported:
            sl-abot :
                Supervised pre-training of A-Bot model using cross
                entropy loss with ground truth answers
            sl-qbot :
                Supervised pre-training of Q-Bot model using cross
                entropy loss with ground truth questions for the
                dialog model and mean squared error loss for image
                feature regression (i.e. image prediction)
            rl-full-QAf :
                RL-finetuning of A-Bot and Q-Bot in a cooperative
                setting where the common reward is the difference
                in mean squared error between the current and
                previous round of Q-Bot's image prediction.
                Annealing: In order to ease in the RL objective,
                fine-tuning starts with first N-1 rounds of SL
                objective and last round of RL objective - the
                number of RL rounds are increased by 1 after
                every epoch until only RL objective is used for
                all rounds of dialog.
        '''
        factRNN = None
        dialogRNN = None
        dialogState = None

        if params['trainMode'] == 'rl-full-QAf' and round >= rlRound and params["AbotMCTS"]:

            factRNN = qBot.encoder.factRNN
            dialogRNN = qBot.encoder.dialogRNN
            dialogState = qBot.encoder.dialogHiddens[-1]

        # Tracking components which require a forward pass
        # A-Bot dialog model
        forwardABot = (params['trainMode'] == 'sl-abot'
                       or (params['trainMode'] == 'rl-full-QAf'
                           and round < rlRound))
        # Q-Bot dialog model
        forwardQBot = (params['trainMode'] == 'sl-qbot'
                       or (params['trainMode'] == 'rl-full-QAf'
                           and round < rlRound))
        # Q-Bot feature regression network
        forwardFeatNet = (forwardQBot or params['trainMode'] == 'rl-full-QAf')

        # Answerer Forward Pass
        if forwardABot:
            # Observe Ground Truth (GT) question
            aBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])
            # Observe GT answer for teacher forcing
            aBot.observe(
                round,
                ans=gtAnswers[:, round],
                ansLens=gtAnsLens[:, round])
            ansLogProbs = aBot.forward()
            # Cross Entropy (CE) Loss for Ground Truth Answers
            aBotLoss += utils.maskedNll(ansLogProbs,
                                        gtAnswers[:, round].contiguous())

        # Questioner Forward Pass (dialog model)
        if forwardQBot:
            # Observe GT question for teacher forcing
            qBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])
            quesLogProbs = qBot.forward()
            # Cross Entropy (CE) Loss for Ground Truth Questions
            qBotLoss += utils.maskedNll(quesLogProbs,
                                        gtQuestions[:, round].contiguous())
            # Observe GT answer for updating dialog history
            qBot.observe(
                round,
                ans=gtAnswers[:, round],
                ansLens=gtAnsLens[:, round])

        # In order to stay true to the original implementation, the feature
        # regression network makes predictions before dialog begins and for
        # the first 9 rounds of dialog. This can be set to 10 if needed.
        MAX_FEAT_ROUNDS = 9

        # Questioner feature regression network forward pass
        if forwardFeatNet and round < MAX_FEAT_ROUNDS:
            # Make an image prediction after each round
            predFeatures = qBot.predictImage()
            featDist = mse_criterion(predFeatures, image)
            featDist = torch.mean(featDist)
            featLoss += featDist

        # Diversity Penalty
        if params["useCosSimilarityLoss"] or params["useHuberLoss"]:

            if params['trainMode'] == 'sl-qbot' or params['trainMode'] == 'rl-full-QAf':
                cur_dialog_hidden = qBot.encoder.dialogHiddens[-1][0]
            elif params['trainMode'] == 'sl-abot':
                cur_dialog_hidden = aBot.encoder.dialogHiddens[-1][0]

            if params["useCosSimilarityLoss"]:
                if round > 0:
                    cos_similarity_loss += utils.cosinePenalty(cur_dialog_hidden, past_dialog_hidden)

            if params["useHuberLoss"]:
                if round > 0:
                    huber_loss += utils.huberPenalty(cur_dialog_hidden, past_dialog_hidden, threshold=0.1)

            if round == 0:

                if params['trainMode'] == 'sl-qbot' or params['trainMode'] == 'rl-full-QAf':
                    past_dialog_hidden = qBot.encoder.dialogHiddens[-1][0]
                elif params['trainMode'] == 'sl-abot':
                    past_dialog_hidden = aBot.encoder.dialogHiddens[-1][0]

            else:
                past_dialog_hidden = cur_dialog_hidden

        # A-Bot and Q-Bot interacting in RL rounds
        if params['trainMode'] == 'rl-full-QAf' and round >= rlRound:

            # Run one round of conversation
            questions, quesLens = qBot.forwardDecode(inference='sample')
            qBot.observe(round, ques=questions, quesLens=quesLens)
            aBot.observe(round, ques=questions, quesLens=quesLens)
            if params["AbotMCTS"]:
                answers, ansLens = aBot.forwardDecode(inference='sample',run_mcts=True)
            else:
                answers, ansLens = aBot.forwardDecode(inference='sample')
            aBot.observe(round, ans=answers, ansLens=ansLens)
            qBot.observe(round, ans=answers, ansLens=ansLens)

            # Q-Bot makes a guess at the end of each round
            predFeatures = qBot.predictImage()

            # Computing reward based on Q-Bot's predicted image
            featDist = mse_criterion(predFeatures, image)
            featDist = torch.mean(featDist,1)

            reward = prevFeatDist.detach() - featDist
            cum_reward = cum_reward + reward.data
            prevFeatDist = featDist
            #TODO this should be an input; ignoring discount factor for now
            if params['rlAbotReward']:
                mean_reward_batch += float(torch.mean(reward))
                aBotRLLoss = aBot.reinforce(reward - baseline)
                qBotRLLoss = qBot.reinforce(reward - baseline)

            rlLoss += torch.mean(aBotRLLoss)
            rlLoss += torch.mean(qBotRLLoss)

    # baseline = (0.95 * baseline) + (0.05 * mean_reward_batch /
    #                                 (params["numRounds"] - rlRound))

    # Loss coefficients
    rlCoeff = params['RLLossCoeff']
    rlLoss = rlLoss * rlCoeff
    featLoss = featLoss * params['featLossCoeff']
    # Averaging over rounds
    qBotLoss = (params['CELossCoeff'] * qBotLoss) / numRounds
    aBotLoss = (params['CELossCoeff'] * aBotLoss) / numRounds
    featLoss = featLoss / numRounds  # / (numRounds+1)
    rlLoss = rlLoss / numRounds

    cos_similarity_loss = (params['CosSimilarityLossCoeff'] * cos_similarity_loss) / numRounds
    # we want to minimize the negative of the Huber Loss.
    huber_loss = -(params["HuberLossCoeff"] * huber_loss)/numRounds

    loss = qBotLoss + aBotLoss + rlLoss + featLoss + cos_similarity_loss + huber_loss
    loss.backward()
    optimizer.step()

    # Tracking a running average of loss
    if runningLoss is None:
        runningLoss = loss.data[0]
    else:
        runningLoss = 0.95 * runningLoss + 0.05 * loss.data[0]

    # Decay learning rate
    if lRate > params['minLRate']:
        for gId, group in enumerate(optimizer.param_groups):
            optimizer.param_groups[gId]['lr'] *= params['lrDecayRate']
        lRate *= params['lrDecayRate']
        if iterId % 10 == 0:  # Plot learning rate till saturation
            viz.linePlot(iterId, lRate, 'learning rate', 'learning rate')

    # RL Annealing: Every epoch after the first, decrease rlRound
    if iterId % numIterPerEpoch == 0 and iterId > 0:
        curEpoch = int(float(iterId) / numIterPerEpoch)
        if curEpoch % params['annealingReduceEpoch'] == 0:
            if params['trainMode'] == 'rl-full-QAf':
                rlRound = max(params["annealingEndRound"], rlRound - 1)
                if rlRound == params["annealingEndRound"]:
                    rlRound = params['numRounds'] - 1
                print('Using rl starting at round {}'.format(rlRound))

    # Print every now and then
    if iterId % 10 == 0:
        end_t = timer()  # Keeping track of iteration(s) time
        curEpoch = float(iterId) / numIterPerEpoch
        timeStamp = strftime('%a %d %b %y %X', gmtime())
        printFormat = '[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.3g]'
        printFormat += '[lr: %.3g]'
        printInfo = [
            timeStamp, curEpoch, iterId, end_t - start_t, loss.data[0], lRate
        ]
        start_t = end_t
        print(printFormat % tuple(printInfo))

        # Update line plots
        if isinstance(aBotLoss, Variable):
            viz.linePlot(iterId, aBotLoss.data[0], 'aBotLoss', 'train CE')
        if isinstance(qBotLoss, Variable):
            viz.linePlot(iterId, qBotLoss.data[0], 'qBotLoss', 'train CE')
        if isinstance(rlLoss, Variable):
            viz.linePlot(iterId, rlLoss.data[0], 'rlLoss', 'Loss')
            avg_reward = torch.mean(cum_reward)
            viz.linePlot(iterId, avg_reward,'rlLoss','Average Cummulative Reward')
        if isinstance(featLoss, Variable):
            viz.linePlot(iterId, featLoss.data[0], 'featLoss',
                         'train FeatureRegressionLoss')
        viz.linePlot(iterId, loss.data[0], 'loss', 'train loss')
        viz.linePlot(iterId, runningLoss, 'loss', 'running train loss')
        if params["useCosSimilarityLoss"]:
            viz.linePlot(iterId, cos_similarity_loss.data[0], "loss","cos similarity loss")
        if params["useHuberLoss"]:
            viz.linePlot(iterId, huber_loss.data[0], "loss", "huber loss")
        if params['trainMode'] == 'rl-full-QAf':
            viz.linePlot(iterId, rlLoss.data[0], "loss", "rl loss")
        if params['trainMode'] == 'sl-qbot' or params['trainMode'] == 'rl-full-QAf':
            viz.linePlot(iterId, featLoss.data[0], "loss", "feat loss")

    # Evaluate every epoch
    if iterId % (numIterPerEpoch // 1) == 0:
        # Keeping track of epochID
        curEpoch = float(iterId) / numIterPerEpoch
        epochId = (1.0 * iterId / numIterPerEpoch) + 1

        # Set eval mode
        if aBot:
            aBot.eval()
        if qBot:
            qBot.eval()

        if params['enableVisdom']:
            # Printing visdom environment name in terminal
            print("Currently on visdom env [%s]" % (params['visdomEnv']))

        # Mapping iteration count to epoch count
        viz.linePlot(iterId, epochId, 'iter x epoch', 'epochs')

        print('Performing validation...')
        if aBot and 'ques' in batch:
            print("aBot Validation:")

            # NOTE: A-Bot validation is slow, so adjust exampleLimit as needed
            rankMetrics = rankABot(
                aBot,
                dataset,
                'val',
                scoringFunction=utils.maskedNll,
                exampleLimit=None,useNDCG=params["useNDCG"])

            for metric, value in rankMetrics.items():
                viz.linePlot(
                    epochId, value, 'val - aBot', metric, xlabel='Epochs')
                print(metric, value)

            if 'logProbsMean' in rankMetrics:
                logProbsMean = params['CELossCoeff'] * rankMetrics[
                    'logProbsMean']
                viz.linePlot(iterId, logProbsMean, 'aBotLoss', 'val CE')
                print("val CE", logProbsMean)

                if params['trainMode'] == 'sl-abot':
                    valLoss = logProbsMean
                    viz.linePlot(iterId, valLoss, 'loss', 'val loss')

        if qBot:
            print("qBot Validation:")
            rankMetrics, roundMetrics = rankQBot(qBot, dataset, 'val')

            for metric, value in rankMetrics.items():
                viz.linePlot(
                    epochId, value, 'val - qBot', metric, xlabel='Epochs')
                print(metric, value)

            viz.linePlot(iterId, epochId, 'iter x epoch', 'epochs')

            if 'logProbsMean' in rankMetrics:
                logProbsMean = params['CELossCoeff'] * rankMetrics[
                    'logProbsMean']
                viz.linePlot(iterId, logProbsMean, 'qBotLoss', 'val CE')
                print("val CE", logProbsMean)

            if 'featLossMean' in rankMetrics:
                featLossMean = params['featLossCoeff'] * (
                    rankMetrics['featLossMean'])
                viz.linePlot(iterId, featLossMean, 'featLoss',
                             'val FeatureRegressionLoss')

            huberLossMean = 0
            if params["useHuberLoss"]:
                huberLossMean = params['HuberLossCoeff'] * (
                    rankMetrics['huberLossMean'])
                viz.linePlot(iterId, huberLossMean, 'val loss',
                             'Huber')
                viz.linePlot(epoch, params['CELossCoeff'] * rankMetrics['logProbsMean'] - rankMetrics['huberLossMean'] * params["HuberLossCoeff"],
                             'val loss','CE - Huber',xlabel='Epochs')
                viz.linePlot(epoch, params['CELossCoeff'] * rankMetrics['logProbsMean'] - rankMetrics['huberLossMean'] * params["HuberLossCoeff"] +
                             rankMetrics['featLossMean'] * params["featLossCoeff"],
                             'val loss','CE - Huber + Feat',xlabel='Epochs')

            cosSimilarityLossMean = 0
            if params["useCosSimilarityLoss"]:
                cosSimilarityLossMean = params['CosSimilarityLossCoeff'] * (
                    rankMetrics['cosSimilarityLossMean'])
                viz.linePlot(iterId, cosSimilarityLossMean, 'val loss',
                             'Cos')
                viz.linePlot(epoch, params['CELossCoeff'] * rankMetrics['logProbsMean'] + rankMetrics['cosSimilarityLossMean'] * params["CosSimilarityLossCoeff"],
                             'val loss','CE + Cos',xlabel='Epochs')
                viz.linePlot(epoch, params['CELossCoeff'] * rankMetrics['logProbsMean'] + rankMetrics['cosSimilarityLossMean'] * params["CosSimilarityLossCoeff"] +
                             rankMetrics['featLossMean'] * params["featLossCoeff"],
                             'val loss','CE + Cos + Feat',xlabel='Epochs')

            if 'logProbsMean' in rankMetrics and 'featLossMean' in rankMetrics:
                if params['trainMode'] == 'sl-qbot':
                    valLoss = logProbsMean + featLossMean

                    if params["useHuberLoss"]:
                        valLoss += huberLossMean
                    if params["useCosSimilarityLoss"]:
                        valLoss += cosSimilarityLossMean

                    viz.linePlot(iterId, valLoss, 'loss', 'val loss')
                    print("valLoss", valLoss)

        if qBot and aBot:
            split = 'val'
            splitName = 'full Val - {}'.format(params['evalTitle'])

            rankMetrics, roundRanks = rankQABots(
                qBot, aBot, dataset, split, beamSize=params['beamSize'])
            for metric, value in rankMetrics.items():
                plotName = splitName + ' - QABots Rank'
                viz.linePlot(epochId, value, plotName, metric, xlabel='Epochs')
                print(metric, value)

    # Save the model after every epoch
    if iterId % numIterPerEpoch == 0:
        params['ckpt_iterid'] = iterId
        params['ckpt_lRate'] = lRate

        if aBot:
            saveFile = os.path.join(params['savePath'],
                                    'abot_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(aBot, optimizer, saveFile, params)
        if qBot:
            saveFile = os.path.join(params['savePath'],
                                    'qbot_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(qBot, optimizer, saveFile, params)

        print("Saving visdom env to disk: {}".format(params["visdomEnv"]))

        viz.save()
