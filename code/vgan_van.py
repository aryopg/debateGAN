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
from generator import GeneratorEncDecTeacherForcingV2 as GeneratorEncDecTeacherForcing
from discriminator import Discriminator

from helpers.utils import llprint

from loss import batchNLLLoss, JSDLoss#, MMDLoss

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
# use_cuda = False
if use_cuda:
    gpu = 0

processed_data_dir = '../data_histo'
train_data_dir = os.path.join(processed_data_dir, 'train')
test_data_dir = os.path.join(processed_data_dir, 'test')
INV_LEXICON_DICTIONARY = pickle.load(open('../data_histo/lexicon-dict-inverse.pkl', 'rb'))

def decode(out):
    ret = []
    out = out.cpu().data.numpy()
    out = np.argmax(out, axis=-1)
    for idx in out:
        sent = []
        for word_id in idx:
            sent.append(INV_LEXICON_DICTIONARY[word_id])
        ret.append(" ".join(sent))
    return ret

def decode_motion(out):
    ret = []
    for idx in out:
        sent = []
        for word_id in idx:
            if word_id == 0: continue
            sent.append(INV_LEXICON_DICTIONARY[word_id])
        ret.append(" ".join(sent))
    return ret

def save_checkpoint(state, filename='last_checkpoint_vanilla_encdec.pth.tar'):
    if os.path.isfile(filename):
        os.remove(filename)
    torch.save(state, filename)

def train(run_name, netG, netD, motion_length, claim_length, embedding_dim, hidden_dim_G, hidden_dim_D, lam, batch_size, epochs, iteration_d, num_words, train_data_dir, test_data_dir):
    jsdloss = JSDLoss()
    criterion = nn.BCELoss()
    print netG
    print netD

    if use_cuda:
        jsdloss = jsdloss.cuda(gpu)
        criterion = criterion.cuda(gpu)
        netD = netD.cuda(gpu)
        netG = netG.cuda(gpu)

    optimizerD = optim.Adam(netD.parameters(), lr=1e-4)#, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4)#, betas=(0.5, 0.9))

    one = torch.FloatTensor([1])
    mone = one * -1
    if use_cuda:
        one = one.cuda(gpu)
        mone = mone.cuda(gpu)

    num_train_data = len(os.listdir(train_data_dir))
    num_test_data = len(os.listdir(test_data_dir))

    training_generator = DataGenerator(lexicon_count=lexicon_count, motion_length=motion_length, claim_length=claim_length, num_data=num_train_data, data_dir=train_data_dir, batch_size=batch_size, shuffle=True)
    test_generator = DataGenerator(lexicon_count=lexicon_count, motion_length=motion_length, claim_length=claim_length, num_data=num_test_data, data_dir=test_data_dir, batch_size=batch_size, shuffle=True)

    start_epoch = 0
    if os.path.isfile('last_checkpoint_vanilla_encdec.pth.tar'):
            print("=> loading checkpoint '{}'".format('last_checkpoint_vanilla_encdec.pth.tar'))
            checkpoint = torch.load('last_checkpoint_vanilla_encdec.pth.tar')
            start_epoch = checkpoint['epoch']
            netD.load_state_dict(checkpoint['D_state_dict'])
            netG.load_state_dict(checkpoint['G_state_dict'])
            optimizerD.load_state_dict(checkpoint['optimizerD'])
            optimizerG.load_state_dict(checkpoint['optimizerG'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format('last_checkpoint_vanilla_encdec.pth.tar', checkpoint['epoch']))

    for iteration in xrange(start_epoch,epochs+1):
        print(" >>> Epoch : %d/%d" % (iteration+1,epochs))
        start_time = time.time()
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for iter_d in xrange(iteration_d):
            training_steps = num_train_data//batch_size
            for training_step in range(training_steps):
                llprint("\rTraining Discriminator : %d/%d ; Training step Discriminator : %d/%d" % (iter_d+1, iteration_d, training_step+1, training_steps))
                _motion, _claim = training_generator.generate().next()
                _claim = np.asarray(_claim)
                real_claim = torch.from_numpy(_claim)
                real_labels = torch.ones(real_claim.size(0),1)

                if use_cuda:
                    real_claim = real_claim.cuda(gpu)
                    real_labels = real_labels.cuda(gpu)
                real_claim_v = autograd.Variable(real_claim)
                real_labels = autograd.Variable(real_labels)

                nn.utils.clip_grad_norm(netD.parameters(), 5.0)
                # for p in netD.parameters():
                #     p.data.clamp_(-5.0, 5.0)
                netD.zero_grad()

                # train with real
                _, D_real = netD(real_claim_v)
                D_real_loss = criterion(D_real, real_labels)

                # train with motion
                _motion = np.asarray(_motion)
                real_motion = torch.LongTensor(_motion.tolist())
                fake_labels = torch.zeros(real_motion.size(0),1)
                if use_cuda:
                    real_motion = real_motion.cuda(gpu)
                    fake_labels = fake_labels.cuda(gpu)
                real_motion_v = autograd.Variable(real_motion, volatile=True)
                fake_labels = autograd.Variable(fake_labels)

                fake = autograd.Variable(netG(real_motion_v, real_claim_v, 0.0).data)
                inputv = fake
                f_fake, D_fake = netD(inputv)

                D_fake_loss = criterion(D_fake, fake_labels)

                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()
                optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = True  # to avoid computation

        print("\nTraining Generator :")
        for iter_g in xrange(iteration_g):
            for training_step in range(num_train_data//batch_size):
                llprint("\rTraining step Generator %d/%d" % (training_step+1, training_steps))

                nn.utils.clip_grad_norm(netG.parameters(), 5.0)
                # for p in netG.parameters():
                #     p.data.clamp_(-5.0, 5.0)
                netG.zero_grad()

                _motion, _claim = training_generator.generate().next()
                _motion = np.asarray(_motion)
                _claim = np.asarray(_claim)
                real_motion = torch.LongTensor(_motion.tolist())
                real_claim_G = torch.from_numpy(np.argmax(_claim, 2))
                real_claim_D = torch.from_numpy(_claim)

                if use_cuda:
                    real_motion = real_motion.cuda(gpu)
                    real_claim_G = real_claim_G.cuda(gpu)
                    real_claim_D = real_claim_D.cuda(gpu)
                real_motion_v = autograd.Variable(real_motion)
                real_claim_G_v = autograd.Variable(real_claim_G)
                real_claim_D_v = autograd.Variable(real_claim_D)

                fake = netG(real_motion_v, real_claim_G_v)
                f_fake, G = netD(fake)
                f_real, _ = netD(real_claim_D_v)

                G_loss = jsdloss(batch_size, f_real, f_fake)
                #G_loss = criterion(G, real_labels)

    	        # G_loss_total = G_loss + G_loss_van

                G_loss.backward()
                optimizerG.step()

        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'.format(iteration, D_loss.cpu().data.numpy()[0], G_loss.cpu().data.numpy()[0]))
        with open('loss_logger.txt', 'a') as out:
            out.write(str(iteration+1) + ',' + str(D_loss.cpu().data.numpy()[0]) + ',' + str(G_loss.cpu().data.numpy()[0]) + '\n')

        if iteration % 5 == 0:
            result_text = open('generated/' + datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S") + ".txt", "w")
            testing_steps = num_test_data//batch_size
            for test_step in range(testing_steps):
                llprint("\rTesting step %d/%d" % (test_step+1, testing_steps))

                motion_test, claim_test = test_generator.generate().next()
                _motion_test = np.asarray(motion_test)
                _claim_test = np.asarray(claim_test)
                real_motion_test = torch.LongTensor(_motion_test.tolist())
                real_claim_test = torch.from_numpy(_claim_test)
                if use_cuda:
                    real_motion_test = real_motion_test.cuda(gpu)
                    real_claim_test = real_claim_test.cuda(gpu)
                real_motion_test_v = autograd.Variable(real_motion_test)
                real_claim_test_v = autograd.Variable(real_claim_test)

                for mot, cla in zip(decode_motion(motion_test), decode(netG(real_motion_test_v, real_claim_test_v, 0.0))):
                    result_text.write("Motion: %s\n" % mot)
                    result_text.write("Generated Claim: %s\n\n" % cla)
        if iteration % 2 == 0:
            save_checkpoint({
                'epoch': iteration + 1,
                'G_state_dict': netG.state_dict(),
                'D_state_dict': netD.state_dict(),
                'optimizerD' : optimizerD.state_dict(),
                'optimizerG' : optimizerG.state_dict(),
            })

if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    motion_length = 20
    claim_length = 20
    embedding_dim = 256
    hidden_dim_G = 128
    hidden_dim_D = 300
    lam = 10
    pretrain_epochs = 1000
    epochs = 1000000
    iteration_d = 5
    iteration_g = 1
    batch_size = 256

    logger = logging.getLogger('eval_textGAN')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('eval_textGAN.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)

    processed_data_dir = '../data_histo'
    generated_data_dir = 'generated'
    pretraining_generated_data_dir = 'pretraining_generated'
    train_data_dir = os.path.join(processed_data_dir, 'train')
    test_data_dir = os.path.join(processed_data_dir, 'test')

    if not os.path.isdir(generated_data_dir):
        os.mkdir(generated_data_dir)

    if not os.path.isdir(pretraining_generated_data_dir):
        os.mkdir(pretraining_generated_data_dir)

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

        netG = GeneratorVan(batch_size, lexicon_count, motion_length, claim_length, hidden_dim_G, embedding_dim)
        netD = Discriminator(batch_size, lexicon_count, claim_length, hidden_dim_D)

        # netD, netG = pretrain(run_name=run_name, netD=netD, netG=netG, motion_length=motion_length, claim_length=claim_length, embedding_dim=embedding_dim, hidden_dim_G=hidden_dim_G, hidden_dim_D=hidden_dim_D, lam=lam, batch_size=batch_size, epochs=pretrain_epochs, num_words=lexicon_count, train_data_dir=train_data_dir, test_data_dir=test_data_dir)
        train(run_name=run_name, netG=netG, netD=netD, motion_length=motion_length, claim_length=claim_length, embedding_dim=embedding_dim, hidden_dim_G=hidden_dim_G, hidden_dim_D=hidden_dim_D, lam=lam, batch_size=batch_size, epochs=epochs, iteration_d=iteration_d, num_words=lexicon_count, train_data_dir=train_data_dir, test_data_dir=test_data_dir)
