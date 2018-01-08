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
import unicodedata
import numpy as np
from shutil import rmtree
from os import listdir, mkdir
from os.path import join, isfile, isdir, dirname, basename, normpath, abspath, exists

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def create_dictionary(motions, claims):
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

    llprint("Creating Dictionary from Motions and Claims ... 0/2")

    # Creating dictionary from motions file
    fobj = csv.reader(open(motions, "rb"), delimiter = '\t')
    for idx, line in enumerate(fobj):
        if idx ==0: continue
        sentence_temp = unicodedata.normalize('NFKD', line[1].decode('utf-8')).encode('ascii', 'ignore')
        sentence_temp = sentence_temp.replace('.', ' .')
        sentence_temp = sentence_temp.replace('?', ' ?')
        sentence_temp = sentence_temp.replace(',', ' ')
        cleaned_motion = sentence_temp.replace('[REF]', '')
        for word in cleaned_motion.split():
            if not word.lower() in lexicons_dict and word.isalpha():
                lexicons_dict[word.lower()] = id_counter
                id_counter += 1

    llprint("\rCreating Dictionary from Motions and Claims ... 1/2")

    # Creating dictionary from claims file
    fobj = csv.reader(open(claims, "rb"), delimiter = '\t')
    for idx, line in enumerate(fobj):
        if idx ==0: continue
        sentence_temp = unicodedata.normalize('NFKD', line[2].decode('utf-8')).encode('ascii', 'ignore')
        sentence_temp = sentence_temp.replace('.', ' .')
        sentence_temp = sentence_temp.replace('?', ' ?')
        sentence_temp = sentence_temp.replace(',', ' ')
        cleaned_claim = sentence_temp.replace('[REF]', '')
        for word in cleaned_claim.split():
            if not word.lower() in lexicons_dict and word.isalpha():
                lexicons_dict[word.lower()] = id_counter
                id_counter += 1

    llprint("\rCreating Dictionary from Motions and Claims ... 2/2")

    print "\rCreating Dictionary from Motions and Claims ... Done!"

    print("\rFound %d unique lexicons" % len(lexicons_dict))

    return lexicons_dict, len(lexicons_dict)

def encode_data(claims, lexicon_dictionary, lexicon_count):
    """
    encodes the dataset into its numeric form given a constructed dictionary

    Parameters:
    ----------
    article_motion_claim_map: dict
        the mappings of motion, claim, and articles
    lexicon_dictionary: dict
        the mappings of unique lexicons

    Returns: tuple (array, int)
        the data in its numeric form, data length
    """

    encoded_data = []
    lengths = []

    fobj = csv.reader(open(claims, "rb"), delimiter = '\t')
    for idx, line in enumerate(fobj):
        if idx == 0: continue
        # try:
        sentence_temp = unicodedata.normalize('NFKD', line[2].decode('utf-8')).encode('ascii', 'ignore')
        sentence_temp = sentence_temp.replace('.', ' .')
        sentence_temp = sentence_temp.replace('?', ' ?')
        sentence_temp = sentence_temp.replace(',', ' ')
        cleaned_claim = sentence_temp.replace('[REF]', '')

        encoded_motion = []
        for word in line[0].split():
            if word.isalpha() or word == '?' or word == '.':
                encoded_motion.append(lexicon_dictionary[word.lower()])

        encoded_claim = []
        for word in cleaned_claim.split():
            if word.isalpha() or word == '?' or word == '.':
                encoded_claim.append(lexicon_dictionary[word.lower()])

        encoded_data.append({
            'input': {
                'motion': encoded_motion
            },
            'output': {
                'encoded': encoded_claim,
                'original': cleaned_claim
            }
        })
        # except:
        #     print(line)
        #     continue

    print "\rEncoding Data ... Done!"
    return encoded_data, len(encoded_data)

if __name__ == '__main__':
    task_dir = dirname(abspath(__file__))
    motions_list = 'dataset/motions.txt'
    claims_list = 'dataset/claims.txt'
    processed_data_dir = join(task_dir, 'data-no-article')
    train_data_dir = join(processed_data_dir, 'train')
    test_data_dir = join(processed_data_dir, 'test')

    if not exists(processed_data_dir):
        mkdir(processed_data_dir)
    if not exists(train_data_dir):
        mkdir(train_data_dir)
    if not exists(test_data_dir):
        mkdir(test_data_dir)

    articles_list = []

    if not isfile(join(processed_data_dir, 'lexicon-dict.pkl')):
        lexicon_dictionary, lexicon_count = create_dictionary(motions=motions_list, claims=claims_list)
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

    lexicon_count = len(lexicon_dictionary)

    if len(listdir(train_data_dir)) <= 0 and len(listdir(test_data_dir)) <= 0:
        encoded_data, num_data = encode_data(claims=claims_list, lexicon_dictionary=lexicon_dictionary, lexicon_count=lexicon_count)

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
        nb_validation_samples = int(0.2 * num_data)
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
