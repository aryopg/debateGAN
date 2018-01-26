from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sys
import csv
try:
    import cPickle as pickle
except:
    import pickle
import getopt
import string
import re
import unicodedata
import numpy as np
from shutil import rmtree
from os import listdir, mkdir
from os.path import join, isfile, isdir, dirname, basename, normpath, abspath, exists

from nltk.tokenize import sent_tokenize, word_tokenize

class Evaluate():
	def __init__(self, candidate, motion, claims):
        super(Evaluate, self).__init__()

        self.chencherry = SmoothingFunction()
 		self.claims_list = '../../dataset/claims.txt'
        self.candidate = candidate
        self.motion = motion
        self.reference = []

        # Make evaluation data
		fobj = csv.reader(open(claims_list, "rb"), delimiter = '\t')
		for idx, line in enumerate(fobj):
		    if idx ==0: continue
		    sentence_temp = unicodedata.normalize('NFKD', line[2].decode('utf-8')).encode('ascii', 'ignore')
		    sentence_temp = sentence_temp.replace('.', '')
		    sentence_temp = sentence_temp.replace('?', '')
		    sentence_temp = sentence_temp.replace('"', '')
		    sentence_temp = sentence_temp.replace('\'', '')
		    sentence_temp = sentence_temp.replace('(', '')
		    sentence_temp = sentence_temp.replace(')', '')
		    sentence_temp = sentence_temp.replace('%', '')
		    sentence_temp = sentence_temp.replace('$', '')
		    sentence_temp = sentence_temp.replace(',', '')
		    cleaned_claim = sentence_temp.replace('[REF]', '')

		    sentence_motion = unicodedata.normalize('NFKD', line[0].decode('utf-8')).encode('ascii', 'ignore')

		    tokenized_claim = word_tokenize(cleaned_claim.lower())
		    if len(tokenized_claim) < 5:
		    	continue
		    else:
		    	self.reference[sentence_motion].append(word_tokenize(cleaned_claim.lower()))

	def BleuScore ():
		# BLEU SCORE
		print('BLEU score : %f'%sentence_bleu(self.reference[self.topic], self.candidate))

		# Unigram
		print('BLEU score for unigram : %f'%sentence_bleu(self.reference[self.topic], self.candidate, weights=(1,0,0,0), smoothing_function=chencherry.method4))

		# 2-Gram
		print('BLEU score for bigram : %f'%sentence_bleu(self.reference[self.topic], self.candidate, weights=(0,1,0,0), smoothing_function=chencherry.method4))

		# 3-Gram
		print('BLEU score for trigram : %f'%sentence_bleu(self.reference[self.topic], self.candidate, weights=(0,0,1,0), smoothing_function=chencherry.method4))

		# 4-Gram
		print('BLEU score for 4-gram : %f'%sentence_bleu(self.reference[self.topic], self.candidate, weights=(0,0,0,1), smoothing_function=chencherry.method4))

		# Combination
		print('Cumulative BLEU score : %f'%sentence_bleu(self.reference[self.topic], self.candidate, weights=(0.25,0.25,0.25,0.25), smoothing_function=chencherry.method4))
