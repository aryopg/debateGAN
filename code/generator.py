import sys
import time
import datetime
import logging
import cPickle

import numpy as np
import random

import torch

# import cv2
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
# use_cuda = False
if use_cuda:
    gpu = 0

class GeneratorEncDec(nn.Module):
    """
        The Generator will first take a motion, and process it. It will go to the embedding layer, and to the GRU layer afterwards.
        After the GRU layer, we can use attention layer similar to Seq2Seq model.

        Flow:
        motion [batch_size, seq_length]
        embedded_motion [batch_size, seq_length, embedding_dim]
        GRU_1_out [batch_size, seq_length, hidden_dim]
        GRU_2_out [batch_size, seq_length, hidden_dim]
        out [batch_size, seq_length, vocab_size]

        out will then be processed by Discriminator
    """
    def __init__(self, batch_size, vocab_size, motion_size, claim_size, hidden_dim, embedding_dim, n_layers=2, dropout_p=0.5):
        super(GeneratorEncDec, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.motion_length = motion_size
        self.claim_length = claim_size

        self.dropout = nn.Dropout(dropout_p)
        self.encoder_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.decoder_gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, self.motion_length)
        self.attn_combine = nn.Linear(hidden_dim * 3, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, motion):
        # ENCODER BLOCK
        # hidden = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        # if use_cuda:
        #     hidden = hidden.cuda()
        # else:
        #     hidden = hidden

        embedded = self.encoder_embeddings(motion) # [batch_size, seq_length, embedding_dim]
        embedded = self.dropout(embedded)

        encoder_output, hidden = self.encoder_gru(embedded, None) #[batch_size, seq_length, hidden_dim]

        encoder_hidden = torch.unsqueeze(hidden.permute(1,0,2)[:,-1,:], 1)

        # DECODER BLOCK
        # initially feed with [SOS] idx
        SOS_token = np.zeros((self.batch_size,1), dtype=np.int32)
        SOS_token = torch.LongTensor(SOS_token.tolist())
        if use_cuda:
            SOS_token = SOS_token.cuda(gpu)
        SOS_token = autograd.Variable(SOS_token)
        y_embedded = self.decoder_embeddings(SOS_token)
        y_embedded = self.dropout(y_embedded)
        # self.dropout(y_embedded)

        attn_weights = F.softmax(self.attn(torch.cat((y_embedded, encoder_hidden), 2)))
        attn_applied = torch.bmm(attn_weights, encoder_output)

        output = torch.cat((y_embedded, attn_applied), 2)
        output = self.attn_combine(output)

        decoder_hidden = encoder_hidden

        # decoder_output_full = []
        # for i in range(self.claim_length):
        #     decoder_output, decoder_hidden = self.decoder_gru(output, decoder_hidden)
        #     decoder_output_full.append(decoder_output)
        # decoder_output = torch.cat(decoder_output_full, 1)


        output = output.permute(1,0,2).squeeze()

        decoder_output = output
        decoder_hidden = decoder_hidden.permute(1,0,2).squeeze()

        decoder_output_full = autograd.Variable(torch.zeros(self.claim_length, self.batch_size, self.hidden_dim))
        decoder_output_full = decoder_output_full.cuda() if use_cuda else decoder_output_full
        # decoder_output_full = []
        for i in range(self.claim_length):
            decoder_hidden = self.decoder_gru_cell(decoder_output, decoder_hidden)
            decoder_output = decoder_hidden
            # decoder_output_full.append(decoder_output)
            decoder_output_full[i, :, :] = decoder_output

        decoder_output_full = decoder_output_full.permute(1,0,2)

        gen_output = self.softmax(self.out(decoder_output_full))

        return gen_output

class GeneratorEncDecTeacherForcing(nn.Module):
    """
        The Generator will first take a motion, and process it. It will go to the embedding layer, and to the GRU layer afterwards.
        After the GRU layer, we can use attention layer similar to Seq2Seq model.

        Flow:
        motion [batch_size, seq_length]
        embedded_motion [batch_size, seq_length, embedding_dim]
        GRU_1_out [batch_size, seq_length, hidden_dim]
        GRU_2_out [batch_size, seq_length, hidden_dim]
        out [batch_size, seq_length, vocab_size]

        out will then be processed by Discriminator
    """
    def __init__(self, batch_size, vocab_size, motion_size, claim_size, hidden_dim, embedding_dim, n_layers=2, dropout_p=0.5):
        super(GeneratorEncDecTeacherForcing, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.motion_length = motion_size
        self.claim_length = claim_size

        self.dropout = nn.Dropout(dropout_p)
        self.encoder_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.decoder_gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, self.motion_length)
        self.attn_combine = nn.Linear(hidden_dim * 3, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()

    def encode(self, motion):
        embedded = self.encoder_embeddings(motion) # [batch_size, seq_length, embedding_dim]
        embedded = self.dropout(embedded)

        encoder_output, hidden = self.encoder_gru(embedded, None) #[batch_size, seq_length, hidden_dim]

        encoder_hidden = torch.unsqueeze(hidden.permute(1,0,2)[:,-1,:], 1)

        return encoder_output, encoder_hidden

    def decode(self, SOS_token, encoder_output, encoder_hidden, target_output, teacher_forcing_ratio=0.8):
        y_embedded = self.decoder_embeddings(SOS_token)
        y_embedded = self.dropout(y_embedded)

        attn_weights = F.softmax(self.attn(torch.cat((y_embedded, encoder_hidden), 2)))
        attn_applied = torch.bmm(attn_weights, encoder_output)

        output = torch.cat((y_embedded, attn_applied), 2)
        output = self.attn_combine(output)
        decoder_hidden = encoder_hidden

        output = output.permute(1,0,2).squeeze()
        decoder_output = output
        decoder_hidden = decoder_hidden.permute(1,0,2).squeeze()

        decoder_output_full = autograd.Variable(torch.zeros(self.claim_length, self.batch_size, self.hidden_dim))
        decoder_output_full = decoder_output_full.cuda() if use_cuda else decoder_output_full

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            target = target_output.permute(1,0)
            for i in range(self.claim_length):
                decoder_hidden = self.decoder_gru_cell(decoder_output, decoder_hidden)
                decoder_output_full[i, :, :] = decoder_hidden
                decoder_output = self.decoder_embeddings(target[i])  # Teacher forcing
                decoder_output = self.dropout(decoder_output)

        else:
            for i in range(self.claim_length):
                decoder_hidden = self.decoder_gru_cell(decoder_output, decoder_hidden)
                decoder_output = decoder_hidden
                decoder_output_full[i, :, :] = decoder_output

        decoder_output_full = decoder_output_full.permute(1,0,2)

        gen_output = self.softmax(self.out(decoder_output_full))

        return gen_output

    def forward(self, motion, target_output, teacher_forcing_ratio=0.8):
        encoder_output, encoder_hidden = self.encode(motion)

        SOS_token = np.zeros((self.batch_size,1), dtype=np.int32)
        SOS_token = torch.LongTensor(SOS_token.tolist())
        if use_cuda:
            SOS_token = SOS_token.cuda(gpu)
        SOS_token = autograd.Variable(SOS_token)

        gen_output = self.decode(SOS_token, encoder_output, encoder_hidden, target_output, teacher_forcing_ratio)

        return gen_output

class GeneratorConvEncDec(nn.Module):
    """
        The Generator will first take a motion, and process it. It will go to the embedding layer, and to the GRU layer afterwards.
        After the GRU layer, we can use attention layer similar to Seq2Seq model.

        Flow:
        motion [batch_size, seq_length]
        embedded_motion [batch_size, seq_length, embedding_dim]
        GRU_1_out [batch_size, seq_length, hidden_dim]
        GRU_2_out [batch_size, seq_length, hidden_dim]
        out [batch_size, seq_length, vocab_size]

        out will then be processed by Discriminator
    """
    def __init__(self, batch_size, vocab_size, motion_size, claim_size, hidden_dim, embedding_dim, n_layers=2, dropout_p=0.5):
        super(GeneratorEncDec, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.motion_length = motion_size
        self.claim_length = claim_size

        self.dropout = nn.Dropout(dropout_p)
        self.encoder_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_conv = nn.Conv1d(embedding_dim, hidden_dim, 3, stride=2)
        self.decoder_gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, self.motion_length)
        self.attn_combine = nn.Linear(hidden_dim * 3, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, motion):
        # ENCODER BLOCK
        # hidden = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        # if use_cuda:
        #     hidden = hidden.cuda()
        # else:
        #     hidden = hidden

        embedded = self.encoder_embeddings(motion) # [batch_size, seq_length, embedding_dim]
        embedded = self.dropout(embedded)

        encoder_output = self.encoder_conv(embedded) #[batch_size, seq_length, hidden_dim]

        encoder_hidden = torch.unsqueeze(encoder_output.permute(1,0,2)[:,-1,:], 1)

        # DECODER BLOCK
        # initially feed with [SOS] idx
        SOS_token = np.zeros((self.batch_size,1), dtype=np.int32)
        SOS_token = torch.LongTensor(SOS_token.tolist())
        if use_cuda:
            SOS_token = SOS_token.cuda(gpu)
        SOS_token = autograd.Variable(SOS_token)
        y_embedded = self.decoder_embeddings(SOS_token)
        y_embedded = self.dropout(y_embedded)
        # self.dropout(y_embedded)

        attn_weights = F.softmax(self.attn(torch.cat((y_embedded, encoder_hidden), 2)))
        attn_applied = torch.bmm(attn_weights, encoder_output)

        output = torch.cat((y_embedded, attn_applied), 2)
        output = self.attn_combine(output)

        decoder_hidden = encoder_hidden

        # decoder_output_full = []
        # for i in range(self.claim_length):
        #     decoder_output, decoder_hidden = self.decoder_gru(output, decoder_hidden)
        #     decoder_output_full.append(decoder_output)
        # decoder_output = torch.cat(decoder_output_full, 1)


        output = output.permute(1,0,2).squeeze()

        decoder_output = output
        decoder_hidden = decoder_hidden.permute(1,0,2).squeeze()

        decoder_output_full = autograd.Variable(torch.zeros(self.claim_length, self.batch_size, self.hidden_dim))
        decoder_output_full = decoder_output_full.cuda() if use_cuda else decoder_output_full
        # decoder_output_full = []
        for i in range(self.claim_length):
            decoder_hidden = self.decoder_gru_cell(decoder_output, decoder_hidden)
            decoder_output = decoder_hidden
            # decoder_output_full.append(decoder_output)
            decoder_output_full[i, :, :] = decoder_output

        decoder_output_full = decoder_output_full.permute(1,0,2)

        gen_output = self.softmax(self.out(decoder_output_full))

        return gen_output

class GeneratorVan(nn.Module):
    """
        The Generator will first take a motion, and process it. It will go to the embedding layer, and to the GRU layer afterwards.
        After the GRU layer, we can use attention layer similar to Seq2Seq model.

        Flow:
        motion [batch_size, seq_length]
        embedded_motion [batch_size, seq_length, embedding_dim]
        GRU_1_out [batch_size, seq_length, hidden_dim]
        GRU_2_out [batch_size, seq_length, hidden_dim]
        out [batch_size, seq_length, vocab_size]

        out will then be processed by Discriminator
    """
    def __init__(self, batch_size, vocab_size, claim_size, hidden_dim, embedding_dim, n_layers=2, dropout_p=0.5):
        super(GeneratorVan, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.claim_length = claim_size

        self.dropout = nn.Dropout(dropout_p)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_dim*2, vocab_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, motion):
        # ENCODER BLOCK
        # hidden = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        # if use_cuda:
        #     hidden = hidden.cuda()
        # else:
        #     hidden = hidden

        embedded = self.embeddings(motion) # [batch_size, seq_length, embedding_dim]
        embedded = self.dropout(embedded)

        encoder_output, hidden = self.gru(embedded, None) #[batch_size, seq_length, hidden_dim]

        gen_output = self.softmax(self.out(encoder_output))

        return gen_output
