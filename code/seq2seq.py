"""
    TODO: pretrain function and model checkpoint
"""

import sys
import time
import datetime
import logging
import cPickle as pickle
import os

import numpy as np

import torch

# import cv2
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from helpers.datagenerator import DataGenerator, FakeDataGenerator

# from generator import GeneratorConvEncDec, GeneratorEncDec, GeneratorEncDecTeacherForcing, GeneratorVan
from generator import GeneratorEncDecTeacherForcingV3 as GeneratorEncDecTeacherForcing
from discriminator import Discriminator

from helpers.utils import llprint

from loss import batchNLLLoss, JSDLoss, MMDLDLoss

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
# use_cuda = False
if use_cuda:
    gpu = 0,1

processed_data_dir = '../data_histo_2'
train_data_dir = os.path.join(processed_data_dir, 'train')
test_data_dir = os.path.join(processed_data_dir, 'test')
INV_LEXICON_DICTIONARY = pickle.load(open('../data_histo_2/lexicon-dict-inverse.pkl', 'rb'))

def decode(out):
    ret = []
    out = out.cpu().data.numpy()
    out = np.argmax(out, axis=-1)
    for idx in out:
        sent = []
        for word_id in idx:
            sent.append(INV_LEXICON_DICTIONARY[word_id])
        ret.append("".join(sent))
    return ret

def decode_motion(out):
    ret = []
    for idx in out:
        sent = []
        for word_id in idx:
            if word_id == 0: continue
            sent.append(INV_LEXICON_DICTIONARY[word_id])
        ret.append("".join(sent))
    return ret

def save_checkpoint(state, filename='last_checkpoint_seq2seq.pth.tar'):
    if os.path.isfile(filename):
        os.remove(filename)
    torch.save(state, filename)

def train(run_name, seq2seq, motion_length, claim_length, embedding_dim, hidden_dim, batch_size, epochs, num_words, train_data_dir, test_data_dir):
    nllloss = batchNLLLoss()
    # bceloss = nn.BCELoss()
    optimizerG = optim.Adam(seq2seq.parameters(), lr=1e-4, betas=(0.5, 0.9))

    if use_cuda:
        nllloss = nllloss.cuda(gpu)
        seq2seq = seq2seq.cuda(gpu)

    num_train_data = len(os.listdir(train_data_dir))
    num_test_data = len(os.listdir(test_data_dir))

    start_epoch = 0
    if os.path.isfile('last_checkpoint_seq2seq.pth.tar'):
        print("=> loading checkpoint '{}'".format('last_checkpoint_seq2seq.pth.tar'))
        checkpoint = torch.load('last_checkpoint_seq2seq.pth.tar')
        start_epoch = checkpoint['epoch']
        seq2seq.load_state_dict(checkpoint['state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
            .format('last_checkpoint_seq2seq.pth.tar', checkpoint['epoch']))

    training_generator = DataGenerator(lexicon_count=lexicon_count, motion_length=motion_length, claim_length=claim_length, num_data=num_train_data, data_dir=train_data_dir, batch_size=batch_size, shuffle=True)
    test_generator = DataGenerator(lexicon_count=lexicon_count, motion_length=motion_length, claim_length=claim_length, num_data=num_test_data, data_dir=test_data_dir, batch_size=batch_size, shuffle=True)

    for iteration in xrange(start_epoch,epochs+1):
        print(" >>> Epoch : %d/%d" % (iteration+1,epochs))
        start_time = time.time()

        print("\nTraining :")
        training_steps = num_train_data//batch_size
        for training_step in range(training_steps):
            llprint("\rTraining step %d/%d" % (training_step+1, training_steps))
            seq2seq.zero_grad()

            _motion, _claim = training_generator.generate().next()
            _motion = np.asarray(_motion)
            _claim = np.asarray(_claim)
            real_motion = torch.LongTensor(_motion.tolist())
            real_claim_G = torch.from_numpy(np.argmax(_claim, 2))
            real_claim_D = torch.from_numpy(_claim).type('torch.LongTensor')

            if use_cuda:
                real_motion = real_motion.cuda(gpu)
                real_claim_G = real_claim_G.cuda(gpu)
                real_claim_D = real_claim_D.cuda(gpu)
            real_motion_v = autograd.Variable(real_motion)
            real_claim_G_v = autograd.Variable(real_claim_G)
            real_claim_D_v = autograd.Variable(real_claim_D)

            fake = seq2seq(real_motion_v, real_claim_G_v)

            G_loss = nllloss(fake, real_claim_D_v, claim_length)

            G_loss.backward()
            G_cost = G_loss
            optimizerG.step()

        print('Iter: {}; G_loss: {:.4}'.format(iteration+1, G_cost.cpu().data.numpy()[0]))
        with open('pretrain_G_loss_logger.txt', 'a') as out:
            out.write(str(iteration+1) + ',' + str(G_cost.cpu().data.numpy()[0]) + '\n')

        if iteration % 20 == 0:
            result_text = open("pretrain_" + datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S") + ".txt", "w")
            testing_steps = num_test_data//batch_size
            for test_step in range(testing_steps):
                llprint("\rTesting step %d/%d" % (test_step+1, testing_steps))

                motion_test, claim_test = test_generator.generate().next()
                _motion_test = np.asarray(motion_test)
                _claim_test = np.asarray(claim_test)
                real_motion_test = torch.LongTensor(_motion_test.tolist())
                real_claim_test = torch.from_numpy(np.argmax(_claim_test, 2))
                if use_cuda:
                    real_motion_test = real_motion_test.cuda(gpu)
                    real_claim_test = real_claim_test.cuda(gpu)
                real_motion_test_v = autograd.Variable(real_motion_test)
                real_claim_test_v = autograd.Variable(real_claim_test)

                for mot, cla in zip(decode_motion(motion_test), decode(seq2seq(real_motion_test_v, real_claim_test_v, 0.0))):
                    result_text.write("Motion: %s\n" % mot)
                    result_text.write("Generated Claim: %s\n\n" % cla)
        if iteration % 10 == 0:
            save_pretrain_checkpoint_G({
                'epoch': iteration + 1,
                'state_dict': seq2seq.state_dict(),
                'optimizer' : optimizerG.state_dict(),
            })

    return seq2seq

if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    motion_length = 75
    claim_length = 32
    embedding_dim = 256
    hidden_dim = 128
    epochs = 1000
    batch_size = 64

    processed_data_dir = '../data_histo_2'
    train_data_dir = os.path.join(processed_data_dir, 'train')
    test_data_dir = os.path.join(processed_data_dir, 'test')

    if not os.path.isfile(os.path.join(processed_data_dir, 'lexicon-dict.pkl')):
        raise IOError("Cannot find %s. Please run preprocess_no_article.py before running train.py" % os.path.join(processed_data_dir, 'lexicon-dict.pkl'))
    else:
        lexicon_dictionary = pickle.load(open(os.path.join(processed_data_dir, 'lexicon-dict.pkl'), 'rb'))

        # append used punctuation to dictionary
        if not '?' in lexicon_dictionary:
            lexicon_dictionary['?'] = lexicon_count
        if not '.' in lexicon_dictionary:
            lexicon_dictionary['.'] = lexicon_count + 1
        if not '-' in lexicon_dictionary:
            lexicon_dictionary['-'] = lexicon_count + 2

        lexicon_count = len(lexicon_dictionary)

        seq2seq = GeneratorEncDecTeacherForcing(batch_size, lexicon_count, motion_length, claim_length, hidden_dim, embedding_dim)

        seq2seq = train(run_name=run_name, seq2seq=seq2seq, motion_length=motion_length, claim_length=claim_length, embedding_dim=embedding_dim, hidden_dim=hidden_dim, batch_size=batch_size, epochs=epochs, num_words=lexicon_count, train_data_dir=train_data_dir, test_data_dir=test_data_dir)
