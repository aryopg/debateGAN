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

from helpers.datagenerator import DataGenerator

from generator import GeneratorConvEncDec, GeneratorEncDec, GeneratorEncDecTeacherForcing, GeneratorVan
from discriminator import Discriminator

from helpers.utils import llprint

from loss import batchNLLLoss, JSDLoss, MMDLDLoss

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
# use_cuda = False
if use_cuda:
    gpu = 0

processed_data_dir = '../data-no-article'
train_data_dir = os.path.join(processed_data_dir, 'train')
test_data_dir = os.path.join(processed_data_dir, 'test')
INV_LEXICON_DICTIONARY = pickle.load(open('../data-no-article/lexicon-dict-inverse.pkl', 'rb'))

def calc_gradient_penalty(batch_size, lam, netD, real_data, fake_data):
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    _, disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients_reshaped = gradients.contiguous().view(gradients.size()[0], -1)
    gradient_penalty = ((gradients_reshaped.norm(2, dim=1) - 1) ** 2).mean() * lam
    # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lam

    return gradient_penalty

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

def save_checkpoint(state, filename='last_checkpoint_wgan.pth.tar'):
    if os.path.isfile(filename):
        os.remove(filename)
    torch.save(state, filename)

def save_pretrain_checkpoint(state, filename='last_checkpoint_vanilla_pretrain_encdec.pth.tar'):
    if os.path.isfile(filename):
        os.remove(filename)
    torch.save(state, filename)

def pretrain(run_name, netD, netG, motion_length, claim_length, embedding_dim, hidden_dim_G, hidden_dim_D, lam, batch_size, epochs, num_words, train_data_dir, test_data_dir):
    # netG = GeneratorEncDec(batch_size, num_words, claim_length, hidden_dim_G, embedding_dim)
    nllloss = batchNLLLoss()
    bceloss = nn.BCELoss()

    if use_cuda:
        nllloss = nllloss.cuda(gpu)
        bceloss = bceloss.cuda(gpu)
        netG = netG.cuda(gpu)
        netD = netD.cuda(gpu)

    optimizerG = optim.Adam(netG.parameters(), lr=1e-4)#, betas=(0.5, 0.9))
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4)#, betas=(0.5, 0.9))

    # one = torch.FloatTensor([1])
    # mone = one * -1
    # if use_cuda:
    #     one = one.cuda(gpu)
    #     mone = mone.cuda(gpu)

    num_train_data = len(os.listdir(train_data_dir))
    num_test_data = len(os.listdir(test_data_dir))

    start_epoch = 0
    if os.path.isfile('last_checkpoint_vanilla_pretrain_encdec_Disc.pth.tar'):
        print("=> loading checkpoint '{}'".format('last_checkpoint_vanilla_pretrain_encdec_Disc.pth.tar'))
        checkpoint = torch.load('last_checkpoint_vanilla_pretrain_encdec_Disc.pth.tar')
        start_epoch = checkpoint['epoch']
        netD.load_state_dict(checkpoint['D_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format('last_checkpoint_vanilla_pretrain_encdec_Disc.pth.tar', checkpoint['epoch']))

    ###########################
    # (1) Pretrain D network  #
    ###########################

    pretraining_generator = FakeDataGenerator(lexicon_count=lexicon_count, motion_length=motion_length, claim_length=claim_length, num_data=num_train_data, data_dir=train_data_dir, batch_size=batch_size, shuffle=True)
    pretraining_test_generator = FakeDataGenerator(lexicon_count=lexicon_count, motion_length=motion_length, claim_length=claim_length, num_data=num_test_data, data_dir=test_data_dir, batch_size=batch_size, shuffle=True)
    netD.batch_size = netD.batch_size*2

    for iteration in xrange(start_epoch,epochs+1):
        print(" >>> Epoch : %d/%d" % (iteration+1,epochs))
        start_time = time.time()

        training_steps = num_train_data//batch_size
        for training_step in range(training_steps):
            llprint("\rPretraining Discriminator : Training step Discriminator %d/%d" % (training_step+1, training_steps))
            netD.zero_grad()

            _claim, _label = pretraining_generator.generate().next()
            _claim = np.asarray(_claim)
            _label = np.asarray(_label)
            claims = torch.from_numpy(_claim)
            labels = torch.from_numpy(_label)

            if use_cuda:
                claims = claims.cuda(gpu)
                labels = labels.cuda(gpu)
            claims_v = autograd.Variable(claims)
            labels_v = autograd.Variable(labels).unsqueeze(1)

            _, D_out = netD(claims_v)

            D_loss = bceloss(D_out, labels_v)

            D_loss.backward()
            D_loss = D_loss
            optimizerD.step()

        testing_steps = num_test_data//batch_size
        D_loss_test = 0
        for testing_step in range(testing_steps):
            llprint("\Testing step Discriminator %d/%d" % (testing_step+1, testing_steps))
            netD.zero_grad()

            _claim, _label = pretraining_test_generator.generate().next()
            _claim = np.asarray(_claim)
            _label = np.asarray(_label)
            claims = torch.from_numpy(_claim)
            labels = torch.from_numpy(_label)

            if use_cuda:
                claims = claims.cuda(gpu)
                labels = labels.cuda(gpu)
            claims_v = autograd.Variable(claims)
            labels_v = autograd.Variable(labels).unsqueeze(1)

            _, D_out = netD(claims_v)

            D_loss_test += bceloss(D_out, labels_v).cpu().data.numpy()[0]

        print('Iter: {}; D_loss: {:.4}'.format(iteration+1, D_loss_test/testing_steps))
        with open('pretrain_D_loss_logger.txt', 'a') as out:
            out.write(str(iteration+1) + ',' + str(D_loss.cpu().data.numpy()[0]) + '\n')

        if iteration % 10 == 0:
            save_pretrain_checkpoint({
                'epoch': iteration + 1,
                'D_state_dict': netD.state_dict(),
                'optimizerD' : optimizerD.state_dict(),
            })

    ###########################
    # (2) Pretrain G network  #
    ###########################

    start_epoch = 0
    if os.path.isfile('last_checkpoint_vanilla_pretrain_encdec_Gen.pth.tar'):
        print("=> loading checkpoint '{}'".format('last_checkpoint_vanilla_pretrain_encdec_Gen.pth.tar'))
        checkpoint = torch.load('last_checkpoint_vanilla_pretrain_encdec_Gen.pth.tar')
        start_epoch = checkpoint['epoch']
        netG.load_state_dict(checkpoint['G_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        print("=> loaded checkpoint '{}' (epoch {})"
            .format('last_checkpoint_vanilla_pretrain_encdec_Gen.pth.tar', checkpoint['epoch']))

    training_generator = DataGenerator(lexicon_count=lexicon_count, motion_length=motion_length, claim_length=claim_length, num_data=num_train_data, data_dir=train_data_dir, batch_size=batch_size, shuffle=True)
    test_generator = DataGenerator(lexicon_count=lexicon_count, motion_length=motion_length, claim_length=claim_length, num_data=num_test_data, data_dir=test_data_dir, batch_size=batch_size, shuffle=True)

    for iteration in xrange(start_epoch,epochs+1):
        print(" >>> Epoch : %d/%d" % (iteration+1,epochs))
        start_time = time.time()

        print("\nPretraining Generator :")
        training_steps = num_train_data//batch_size
        for training_step in range(training_steps):
            llprint("\rTraining step Generator %d/%d" % (training_step+1, training_steps))
            netG.zero_grad()

            _motion, _claim = training_generator.generate().next()
            _motion = np.asarray(_motion)
            _claim = np.asarray(_claim)
            real_motion = torch.LongTensor(_motion.tolist())
            real_claim = torch.from_numpy(_claim).type('torch.LongTensor')

            if use_cuda:
                real_motion = real_motion.cuda(gpu)
            real_motion_v = autograd.Variable(real_motion)

            if use_cuda:
                real_claim = real_claim.cuda(gpu)
            real_claim_v = autograd.Variable(real_claim)

            fake = netG(real_motion_v)

            G_loss_1 = nllloss(fake, real_claim_v, claim_length)

            G_loss_1.backward()
            G_cost = G_loss_1
            optimizerG.step()

        print('Iter: {}; G_loss: {:.4}'.format(iteration+1, G_cost.cpu().data.numpy()[0]))
        with open('pretrain_G_loss_logger.txt', 'a') as out:
            out.write(str(iteration+1) + ',' + str(G_cost.cpu().data.numpy()[0]) + '\n')

        if iteration % 100 == 0:
            result_text = open("pretrain_" + datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S") + ".txt", "w")
            testing_steps = num_test_data//batch_size
            for test_step in range(testing_steps):
                llprint("\rTesting step %d/%d" % (test_step+1, testing_steps))

                motion_test, claim_test = test_generator.generate().next()
                _motion_test = np.asarray(motion_test)
                real_motion_test = torch.LongTensor(_motion_test.tolist())
                if use_cuda:
                    real_motion_test = real_motion_test.cuda(gpu)
                real_motion_test_v = autograd.Variable(real_motion_test)

                for mot, cla in zip(decode_motion(motion_test), decode(netG(real_motion_test_v))):
                    result_text.write("Motion: %s\n" % mot)
                    result_text.write("Generated Claim: %s\n\n" % cla)
        if iteration % 10 == 0:
            save_pretrain_checkpoint({
                'epoch': iteration + 1,
                'G_state_dict': netG.state_dict(),
                'optimizerG' : optimizerG.state_dict(),
            })

    return netD, netG

def train(run_name, netG, netD, motion_length, claim_length, embedding_dim, hidden_dim_G, hidden_dim_D, lam, batch_size, epochs, iteration_d, num_words, train_data_dir, test_data_dir):
    # netG = GeneratorEncDec(batch_size, num_words, claim_length, hidden_dim_G, embedding_dim)
    jsdloss = JSDLoss()
    cosine = nn.CosineSimilarity()
    print netG
    print netD

    if use_cuda:
        jsdloss = jsdloss.cuda(gpu)
        cosine = cosine.cuda(gpu)
        netD = netD.cuda(gpu)
        netG = netG.cuda(gpu)

    optimizerD = optim.Adam(netD.parameters(), lr=5e-5, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=5e-5, betas=(0.5, 0.9))

    one = torch.FloatTensor([1])
    mone = one * -1
    if use_cuda:
        one = one.cuda(gpu)
        mone = mone.cuda(gpu)

    num_train_data = len(os.listdir(train_data_dir))
    num_test_data = len(os.listdir(test_data_dir))

    training_generator = DataGenerator(lexicon_count=lexicon_count, motion_length=motion_length, claim_length=claim_length, num_data=num_train_data, data_dir=train_data_dir, batch_size=batch_size, shuffle=True)
    test_generator = DataGenerator(lexicon_count=lexicon_count, motion_length=motion_length, claim_length=claim_length, num_data=num_test_data, data_dir=test_data_dir, batch_size=batch_size, shuffle=True)

    start_epoch=0
    if os.path.isfile('last_checkpoint_wgan.pth.tar'):
            print("=> loading checkpoint '{}'".format('last_checkpoint_wgan.pth.tar'))
            checkpoint = torch.load('last_checkpoint_wgan.pth.tar')
            start_epoch = checkpoint['epoch']
            netD.load_state_dict(checkpoint['D_state_dict'])
            netG.load_state_dict(checkpoint['G_state_dict'])
            optimizerD.load_state_dict(checkpoint['optimizerD'])
            optimizerG.load_state_dict(checkpoint['optimizerG'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format('last_checkpoint_wgan.pth.tar', checkpoint['epoch']))

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
                _motion = np.asarray(_motion)
                _claim = np.asarray(_claim)
                real_claim = torch.from_numpy(_claim)
                real_motion = torch.LongTensor(_motion.tolist())

                if use_cuda:
                    real_claim = real_claim.cuda(gpu)
                    real_motion = real_motion.cuda(gpu)
                real_claim_v = autograd.Variable(real_claim)
                real_motion_v = autograd.Variable(real_motion, volatile=True)

                nn.utils.clip_grad_norm(netD.parameters(), 5)
                netD.zero_grad()

                # train with real
                f_real, D_real = netD(real_claim_v)
                D_real = D_real.mean()
                D_real.backward(mone, retain_graph=True)

                # train with motion
                fake = autograd.Variable(netG(real_motion_v, real_claim_v, 0.0).data)
                inputv = fake
                f_fake, D_fake = netD(inputv)
                D_fake = D_fake.mean()
                D_fake.backward(one, retain_graph=True)

                D_jsd = jsdloss(batch_size, f_fake, f_real).mean()
                D_jsd.backward(mone)

                # train with gradient penalty
                gradient_penalty = calc_gradient_penalty(batch_size, lam, netD, real_claim_v.data, fake.data)# + cosine(f_fake, f_real).mean()
                # print gradient_penalty
                gradient_penalty.backward()

                D_cost = D_fake - D_real + gradient_penalty
                Wasserstein_D = D_real - D_fake
                optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation

        print("\nTraining Generator :")
        for training_step in range(num_train_data//batch_size):
            llprint("\rTraining step Generator %d/%d" % (training_step+1, training_steps))

            nn.utils.clip_grad_norm(netG.parameters(), 5)
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

            # G_loss = G.mean()
            G_loss = G.mean() + jsdloss(batch_size, f_fake, f_real).mean()

            G_loss.backward(mone)
            G_cost = -G_loss
            optimizerG.step()

        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; Wasserstein_D: {:.4}'.format(iteration, D_cost.cpu().data.numpy()[0], G_cost.cpu().data.numpy()[0], Wasserstein_D.cpu().data.numpy()[0]))
        with open('loss_logger.txt', 'a') as out:
            out.write(str(iteration+1) + ',' + str(D_cost.cpu().data.numpy()[0]) + ',' + str(G_cost.cpu().data.numpy()[0]) + ',' + str(Wasserstein_D.cpu().data.numpy()[0]) + '\n')

        if iteration % 5 == 0:
            result_text = open(datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S") + ".txt", "w")
            testing_steps = num_test_data//batch_size
            for test_step in range(testing_steps):
                llprint("\rTesting step %d/%d" % (test_step+1, testing_steps))

                motion_test, claim_test = test_generator.generate().next()
                _motion_test = np.asarray(motion_test)
                real_motion_test = torch.LongTensor(_motion_test.tolist())
                if use_cuda:
                    real_motion_test = real_motion_test.cuda(gpu)
                real_motion_test_v = autograd.Variable(real_motion_test)

                for mot, cla in zip(decode_motion(motion_test), decode(netG(real_motion_test_v))):
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
    pretrain_epochs = 10000
    epochs = 1000000
    iteration_d = 5
    batch_size = 64

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

    processed_data_dir = '../data-no-article'
    train_data_dir = os.path.join(processed_data_dir, 'train')
    test_data_dir = os.path.join(processed_data_dir, 'test')
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

        netG = GeneratorEncDecTeacherForcing(batch_size, lexicon_count, motion_length, claim_length, hidden_dim_G, embedding_dim)
        netD = Discriminator(batch_size, lexicon_count, claim_length, hidden_dim_D)

        # netD, netG = pretrain(run_name=run_name, netD=netD, netG=netG, motion_length=motion_length, claim_length=claim_length, embedding_dim=embedding_dim, hidden_dim_G=hidden_dim_G, hidden_dim_D=hidden_dim_D, lam=lam, batch_size=batch_size, epochs=pretrain_epochs, num_words=lexicon_count, train_data_dir=train_data_dir, test_data_dir=test_data_dir)
        train(run_name=run_name, netG=netG, netD=netD, motion_length=motion_length, claim_length=claim_length, embedding_dim=embedding_dim, hidden_dim_G=hidden_dim_G, hidden_dim_D=hidden_dim_D, lam=lam, batch_size=batch_size, epochs=epochs, iteration_d=iteration_d, num_words=lexicon_count, train_data_dir=train_data_dir, test_data_dir=test_data_dir)
