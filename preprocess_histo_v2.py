"""

TODO:

1. word tokenizer for all articles, motions, claims
2. save to pickle word index
3. train - val - test split

"""

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

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def create_dictionary(articles, motions, claims, evidence):
    """
    creates a dictionary of unique lexicons in the dataset and their mapping to numbers

    Parameters:
    ----------
    articles: list
        the list of articles to scan through
    motions: string
        the path to the motions file to scan through
    claims: string
        the path to the claims file to scan through
    evidence: string
        the path to the evidence file to scan through

    Returns: dict
        the constructed dictionary of lexicons
    """

    lexicons_dict = {}
    id_counter = 0
    lexicons_dict['<SOS>'] = id_counter
    id_counter += 1
    lexicons_dict['<EOS>'] = id_counter
    id_counter += 1
    lexicons_dict['<PAD>'] = id_counter
    id_counter += 1
    lexicons_dict['<UNK>'] = id_counter
    id_counter += 1

    histogram = "word_histogram_new.csv"
    lexicons_dict_histo = {}
    id_counter_histo = 0
    lexicons_dict_histo['?'] = id_counter_histo
    lexicons_dict_histo['.'] = id_counter_histo + 1
    lexicons_dict_histo['-'] = id_counter_histo + 2
    lexicons_dict_histo['$'] = id_counter_histo + 3
    lexicons_dict_histo['%'] = id_counter_histo + 4

    print("Creating dict from histogram")
    fobj = csv.reader(open(histogram, "rb"), delimiter = ',')
    for idx, line in enumerate(fobj):
        if idx == 0: continue
        if (int(line[1]) > 6 ):
            # print line[1]
            word = unicodedata.normalize('NFKD', line[0].decode('utf-8')).encode('ascii', 'ignore')
            if not word.lower() in lexicons_dict_histo and word.isalnum():
                lexicons_dict_histo[word.lower()] = id_counter_histo
                id_counter_histo += 1

    llprint("Creating dict from histogram DONE!")
    llprint("\rCreating Dictionary from Articles, Motions, Claims, and Evidence ... 0/4")

    for indx, filename in enumerate(articles):
        with open(filename, 'r') as fobj:
            raw = fobj.read()
            raw = unicodedata.normalize('NFKD', raw.decode('utf-8')).encode('ascii', 'ignore')
            raw_sentences = sent_tokenize(raw)
            for sentence in raw_sentences:
                sentence = sentence.replace("\n", "")
                sentence = sentence.replace('.', ' .')
                sentence = sentence.replace('"', '')
                sentence = sentence.replace('\'', '')
                sentence = sentence.replace('(', '')
                sentence = sentence.replace(')', '')
                sentence = sentence.replace('%', ' %')
                sentence = sentence.replace('$', '$ ')
                sentence = sentence.replace('?', ' ?')
                sentence = sentence.replace(',', ' ')
                sentence = sentence.replace('[REF]', '')
                sentence = sentence.replace('  . ', '')
                cleaned_sentence = sentence.strip()
                if cleaned_sentence:
                    for idx, word in enumerate(word_tokenize(cleaned_sentence)):
                        if idx <= 10:
                            word = re.compile('[%s]' % re.escape(string.punctuation)).sub('', word)
                            if not word.lower() in lexicons_dict and word.isalpha() and word in lexicons_dict_histo:
                                lexicons_dict[word.lower()] = id_counter
                                id_counter += 1

    llprint("Creating Dictionary from Articles, Motions, Claims, and Evidence ... 1/4")

    # Creating dictionary from motions file
    fobj = csv.reader(open(motions, "rb"), delimiter = '\t')
    for idx, line in enumerate(fobj):
        if idx ==0: continue
        sentence_temp = unicodedata.normalize('NFKD', line[1].decode('utf-8')).encode('ascii', 'ignore')
        sentence_temp = sentence_temp.replace('.', ' .')
        sentence_temp = sentence_temp.replace('"', '')
        sentence_temp = sentence_temp.replace('\'', '')
        sentence_temp = sentence_temp.replace('(', '')
        sentence_temp = sentence_temp.replace(')', '')
        sentence_temp = sentence_temp.replace('%', ' %')
        sentence_temp = sentence_temp.replace('$', '$ ')
        sentence_temp = sentence_temp.replace('?', ' ?')
        sentence_temp = sentence_temp.replace(',', ' ')
        cleaned_motion = sentence_temp.replace('[REF]', '')
        for idx, word in enumerate(word_tokenize(cleaned_motion)):
            if idx <= 25:
                word = re.compile('[%s]' % re.escape(string.punctuation)).sub('', word)
                if not word.lower() in lexicons_dict and word.isalpha() and word in lexicons_dict_histo:
                    lexicons_dict[word.lower()] = id_counter
                    id_counter += 1

    llprint("Creating Dictionary from Articles, Motions, Claims, and Evidence ... 2/4")

    # Creating dictionary from claims file
    fobj = csv.reader(open(claims, "rb"), delimiter = '\t')
    for idx, line in enumerate(fobj):
        if idx ==0: continue
        sentence_temp = unicodedata.normalize('NFKD', line[2].decode('utf-8')).encode('ascii', 'ignore')
        sentence_temp = sentence_temp.replace('.', ' .')
        sentence_temp = sentence_temp.replace('?', ' ?')
        sentence_temp = sentence_temp.replace('"', '')
        sentence_temp = sentence_temp.replace('\'', '')
        sentence_temp = sentence_temp.replace('(', '')
        sentence_temp = sentence_temp.replace(')', '')
        sentence_temp = sentence_temp.replace('%', ' %')
        sentence_temp = sentence_temp.replace('$', '$ ')
        sentence_temp = sentence_temp.replace(',', ' ')
        cleaned_claim = sentence_temp.replace('[REF]', '')
        for idx, word in enumerate(word_tokenize(cleaned_claim)):
            if idx <= 10:
                word = re.compile('[%s]' % re.escape(string.punctuation)).sub('', word)
                if not word.lower() in lexicons_dict and word.isalpha() and word in lexicons_dict_histo:
                    lexicons_dict[word.lower()] = id_counter
                    id_counter += 1

    llprint("Creating Dictionary from Articles, Motions, Claims, and Evidence ... 3/4")

    fobj = csv.reader(open(evidence, "rb"), delimiter = '\t')
    for idx, line in enumerate(fobj):
        sentence_temp = unicodedata.normalize('NFKD', line[2].decode('utf-8')).encode('ascii', 'ignore')
        sentence_temp = sentence_temp.replace('.', ' .')
        sentence_temp = sentence_temp.replace('?', ' ?')
        sentence_temp = sentence_temp.replace(',', ' ')
        sentence_temp = sentence_temp.replace('"', '')
        sentence_temp = sentence_temp.replace('\'', '')
        sentence_temp = sentence_temp.replace('(', '')
        sentence_temp = sentence_temp.replace(')', '')
        sentence_temp = sentence_temp.replace('%', ' %')
        sentence_temp = sentence_temp.replace('$', '$ ')
        cleaned_evidence = sentence_temp.replace('[REF]', '')
        cleaned_evidence = cleaned_evidence.replace('[REF', '')
        cleaned_evidence = cleaned_evidence.replace('[STUDY]', '')
        cleaned_evidence = cleaned_evidence.replace('[ANECDOTAL]', '')
        cleaned_evidence = cleaned_evidence.replace('[EXPERT]', '')
        cleaned_evidence = cleaned_evidence.replace('[STUDY, EXPERT]', '')
        cleaned_evidence = cleaned_evidence.replace('[EXPERT]', '')
        for idx, word in enumerate(word_tokenize(cleaned_evidence)):
            if idx <= 10:
                word = re.compile('[%s]' % re.escape(string.punctuation)).sub('', word)
                if not word.lower() in lexicons_dict and word.isalpha() and word in lexicons_dict_histo:
                    lexicons_dict[word.lower()] = id_counter
                    id_counter += 1

    print "\rCreating Dictionary from Motions and Claims ... Done!"

    print("\rFound %d unique lexicons" % len(lexicons_dict))

    return lexicons_dict, len(lexicons_dict)

def compile_data_evidence_motion(evidence_list):
    motion_evidence_map = {}

    fobj = csv.reader(open(evidence_list, "rb"), delimiter = '\t')
    for idx, line in enumerate(fobj):
        sentence_temp = unicodedata.normalize('NFKD', line[2].decode('utf-8')).encode('ascii', 'ignore')
        sentence_temp = sentence_temp.replace('.', ' .')
        sentence_temp = sentence_temp.replace('?', ' ?')
        sentence_temp = sentence_temp.replace(',', ' ')
        sentence_temp = sentence_temp.replace('"', '')
        sentence_temp = sentence_temp.replace('\'', '')
        sentence_temp = sentence_temp.replace('(', '')
        sentence_temp = sentence_temp.replace(')', '')
        sentence_temp = sentence_temp.replace('%', ' %')
        sentence_temp = sentence_temp.replace('$', '$ ')
        cleaned_evidence = sentence_temp.replace('[REF]', '')
        cleaned_evidence = cleaned_evidence.replace('[REF', '')
        cleaned_evidence = cleaned_evidence.replace('[STUDY]', '')
        cleaned_evidence = cleaned_evidence.replace('[ANECDOTAL]', '')
        cleaned_evidence = cleaned_evidence.replace('[EXPERT]', '')
        cleaned_evidence = cleaned_evidence.replace('[STUDY, EXPERT]', '')
        cleaned_evidence = cleaned_evidence.replace('[EXPERT]', '')
        if cleaned_evidence and len(word_tokenize(cleaned_evidence)) > 4:
            motion_evidence_map[cleaned_evidence] = line[0]

    return motion_evidence_map

def compile_data_claim_motion(claims_list):
    motion_claim_map = {}

    fobj = csv.reader(open(claims_list, "rb"), delimiter = '\t')
    for idx, line in enumerate(fobj):
        if idx == 0: continue
        sentence_temp = unicodedata.normalize('NFKD', line[2].decode('utf-8')).encode('ascii', 'ignore')
        sentence_temp = sentence_temp.replace('.', ' .')
        sentence_temp = sentence_temp.replace('?', ' ?')
        sentence_temp = sentence_temp.replace('"', '')
        sentence_temp = sentence_temp.replace('\'', '')
        sentence_temp = sentence_temp.replace('(', '')
        sentence_temp = sentence_temp.replace(')', '')
        sentence_temp = sentence_temp.replace('%', ' %')
        sentence_temp = sentence_temp.replace('$', '$ ')
        sentence_temp = sentence_temp.replace(',', ' ')
        cleaned_claim = sentence_temp.replace('[REF]', '')
        if cleaned_claim and len(word_tokenize(cleaned_claim)) > 4:
            motion_claim_map[cleaned_claim] = line[0]

    return motion_claim_map

def compile_data_article_motion(articles_file):
    motion_article_map = {}

    fobj = csv.reader(open(articles_file, "rb"), delimiter = '\t')
    correlated_articles = []
    for idx, line in enumerate(fobj):
        if idx == 0: continue
        with open('dataset/articles/clean_' + str(line[2]) + '.txt', 'r') as article:
            raw = article.read()
            # print raw
            raw = unicodedata.normalize('NFKD', raw.decode('utf-8')).encode('ascii', 'ignore')
            raw_sentences = sent_tokenize(raw)
            for sentence in raw_sentences:
                sentence = sentence.replace("\n", "")
                sentence = sentence.replace('.', ' .')
                sentence = sentence.replace('"', '')
                sentence = sentence.replace('\'', '')
                sentence = sentence.replace('(', '')
                sentence = sentence.replace(')', '')
                sentence = sentence.replace('%', ' %')
                sentence = sentence.replace('$', '$ ')
                sentence = sentence.replace('?', ' ?')
                sentence = sentence.replace(',', ' ')
                sentence = sentence.replace('[REF]', '')
                sentence = sentence.replace('  . ', '')
                cleaned_sentence = sentence.strip()
                if cleaned_sentence and len(word_tokenize(cleaned_sentence)) > 4:
                    motion_article_map[cleaned_sentence] = line[0]

    return motion_article_map

def encode_data(motion_evidence_map, motion_claim_map, motion_article_map, lexicon_dictionary, lexicon_count):
    encoded_data = []
    lengths = []

    for key, value in motion_evidence_map.iteritems():
        encoded_motion = []
        for idx, word in enumerate(word_tokenize(value)):
            if idx <= 25:
                if word in lexicon_dictionary :
                    word = re.compile('[%s]' % re.escape(string.punctuation)).sub('', word)
                    if word.isalpha() or word == '?' or word == '.':
                        encoded_motion.append(lexicon_dictionary[word.lower()])
                else:
                    encoded_motion.append(lexicon_dictionary['<UNK>'])

        encoded_sentence = []
        if len(word_tokenize(key)) > 4:
            for idx, word in enumerate(word_tokenize(key)):
                if idx <= 10:
                    if word in lexicon_dictionary:
                        word = re.compile('[%s]' % re.escape(string.punctuation)).sub('', word)
                        if word.isalpha() or word == '?' or word == '.':
                            encoded_sentence.append(lexicon_dictionary[word.lower()])
                    else:
                        encoded_sentence.append(lexicon_dictionary['<UNK>'])

        encoded_data.append({
            'input': {
                'encoded': encoded_motion,
                'ori': value
            },
            'output': {
                'encoded': encoded_sentence,
                'ori': key
            }
        })

    for key, value in motion_claim_map.iteritems():
        encoded_motion = []
        for idx, word in enumerate(word_tokenize(value)):
            if idx <= 25:
                if word in lexicon_dictionary:
                    word = re.compile('[%s]' % re.escape(string.punctuation)).sub('', word)
                    if word.isalpha() or word == '?' or word == '.':
                        encoded_motion.append(lexicon_dictionary[word.lower()])
                else:
                    encoded_motion.append(lexicon_dictionary['<UNK>'])
        encoded_sentence = []
        if len(word_tokenize(key)) > 4:
            for idx, word in enumerate(word_tokenize(key)):
                if idx <= 10:
                    if word in lexicon_dictionary:
                        word = re.compile('[%s]' % re.escape(string.punctuation)).sub('', word)
                        if word.isalpha() or word == '?' or word == '.':
                            encoded_sentence.append(lexicon_dictionary[word.lower()])
                    else:
                        encoded_sentence.append(lexicon_dictionary['<UNK>'])

        encoded_data.append({
            'input': {
                'encoded': encoded_motion,
                'ori': value
            },
            'output': {
                'encoded': encoded_sentence,
                'ori': key
            }
        })

    for key, value in motion_article_map.iteritems():
        encoded_motion = []
        for idx, word in enumerate(word_tokenize(value)):
            if idx <= 25:
                if word in lexicon_dictionary:
                    word = re.compile('[%s]' % re.escape(string.punctuation)).sub('', word)
                    if word.isalpha() or word == '?' or word == '.':
                        encoded_motion.append(lexicon_dictionary[word.lower()])
                else:
                    encoded_motion.append(lexicon_dictionary['<UNK>'])
        encoded_sentence = []
        if len(word_tokenize(key)) > 4:
            for idx, word in enumerate(word_tokenize(key)):
                if idx <= 10:
                    if word in lexicon_dictionary:
                        word = re.compile('[%s]' % re.escape(string.punctuation)).sub('', word)
                        if word.isalpha() or word == '?' or word == '.':
                            encoded_sentence.append(lexicon_dictionary[word.lower()])
                    else:
                        encoded_sentence.append(lexicon_dictionary['<UNK>'])

        encoded_data.append({
            'input': {
                'encoded': encoded_motion,
                'ori': value
            },
            'output': {
                'encoded': encoded_sentence,
                'ori': key
            }
        })

    print "\rEncoding Data ... Done!"
    return encoded_data, len(encoded_data)

if __name__ == '__main__':
    task_dir = dirname(abspath(__file__))
    articles_dir = 'dataset/articles'
    articles_file = 'dataset/articles.txt'
    motions_list = 'dataset/motions.txt'
    claims_list = 'dataset/claims.txt'
    evidence_list = 'dataset/evidence.txt'
    processed_data_dir = join(task_dir, 'data_histo_2')
    train_data_dir = join(processed_data_dir, 'train')
    test_data_dir = join(processed_data_dir, 'test')

    if not exists(processed_data_dir):
        mkdir(processed_data_dir)
    if not exists(train_data_dir):
        mkdir(train_data_dir)
    if not exists(test_data_dir):
        mkdir(test_data_dir)

    articles_list = []

    if not exists(join(task_dir, 'data_new_2')):
        mkdir(join(task_dir, 'data_new_2'))

    for entryname in listdir(articles_dir):
        entry_path = join(articles_dir, entryname)
        if isfile(entry_path):
            articles_list.append(entry_path)

    if not isfile(join(processed_data_dir, 'lexicon-dict.pkl')):
        lexicon_dictionary, lexicon_count = create_dictionary(articles=articles_list, motions=motions_list, claims=claims_list, evidence=evidence_list)
    else:
        lexicon_dictionary = pickle.load(open(join(processed_data_dir, 'lexicon-dict.pkl'), 'rb'))
        lexicon_count = len(lexicon_dictionary)

    # append used punctuation to dictionary
    if not '?' in lexicon_dictionary:
        lexicon_dictionary['?'] = lexicon_count
    if not '.' in lexicon_dictionary:
        lexicon_dictionary['.'] = lexicon_count + 1
    if not '-' in lexicon_dictionary:
        lexicon_dictionary['-'] = lexicon_count + 2
    if not '$' in lexicon_dictionary:
        lexicon_dictionary['$'] = lexicon_count + 3
    if not '%' in lexicon_dictionary:
        lexicon_dictionary['%'] = lexicon_count + 4

    if not '?' in lexicon_dictionary:
        lexicon_dictionary['?'] = lexicon_count
    if not '.' in lexicon_dictionary:
        lexicon_dictionary['.'] = lexicon_count + 1
    if not '-' in lexicon_dictionary:
        lexicon_dictionary['-'] = lexicon_count + 2
    if not '$' in lexicon_dictionary:
        lexicon_dictionary['$'] = lexicon_count + 3
    if not '%' in lexicon_dictionary:
        lexicon_dictionary['%'] = lexicon_count + 4

    lexicon_count = len(lexicon_dictionary)

    motion_evidence_map = compile_data_evidence_motion(evidence_list)
    motion_claim_map = compile_data_claim_motion(claims_list)
    motion_article_map = compile_data_article_motion(articles_file)

    if len(listdir(train_data_dir)) <= 0 and len(listdir(test_data_dir)) <= 0:
        encoded_data, num_data = encode_data(motion_evidence_map=motion_evidence_map,
                                            motion_claim_map=motion_claim_map,
                                            motion_article_map=motion_article_map,
                                            lexicon_dictionary=lexicon_dictionary,
                                            lexicon_count=lexicon_count)

        print "Total Number of Data: %d" % (num_data)

    print("Saving processed data to disk ... ")

    if not isfile(join(processed_data_dir, 'lexicon-dict.pkl')):
        pickle.dump(lexicon_dictionary, open(join(processed_data_dir, 'lexicon-dict.pkl'), 'wb'))
    if not isfile(join(processed_data_dir, 'lexicon-dict-inverse.pkl')):
        inv_lexicon_dictionary = {value: key for key, value in lexicon_dictionary.iteritems()}
        pickle.dump(inv_lexicon_dictionary, open(join(processed_data_dir, 'lexicon-dict-inverse.pkl'), 'wb'))

    if len(listdir(train_data_dir)) <= 0 and len(listdir(test_data_dir)) <= 0:
        seed = 7
        np.random.seed(seed)
        np.random.shuffle(encoded_data)
        # encoded_data = encoded_data[indices]
        nb_validation_samples = int(0.1 * num_data)
        data_train = encoded_data[:-nb_validation_samples]
        data_test = encoded_data[-nb_validation_samples:]

        llprint("\rSaving Input-Output Training Data ... 0/%d" % len(data_train))
        for idx, data in enumerate(data_train):
            pickle.dump(data, open(join(train_data_dir, basename('encoded_input_output') + '_' + str(idx+1) + '.pkl'), 'wb'))
            llprint("\rSaving Input-Output Training Data ... %d/%d" % (idx + 1, len(data_train)))

        print("\rSaving Input-Output Training Data ... Done!")

        llprint("\rSaving Input-Output Testing Data ... 0/%d" % len(data_test))
        for idx, data in enumerate(data_test):
            pickle.dump(data, open(join(test_data_dir, basename('encoded_input_output') + '_' + str(idx+1) + '.pkl'), 'wb'))
            llprint("\rSaving Input-Output Testing Data ... %d/%d" % (idx + 1, len(data_test)))

        print("\rSaving Input-Output Testing Data ... Done!")

    llprint("Done!\n")
