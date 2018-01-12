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

from nltk.tokenize import sent_tokenize, word_tokenize

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def create_dictionary(file):
    """
    creates a dictionary of unique lexicons in the dataset and their mapping to numbers

    Parameters:
    ----------
    file: string
        path to the csv histogram file

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

    print("Creating Dictionary ...")

    fobj = csv.reader(open(file, "rb"), delimiter = ',')
    for idx, line in enumerate(fobj):
        if idx == 0: continue
        if (int(line[1]) >=10 ):
            # print line[1]
            word = unicodedata.normalize('NFKD', line[0].decode('utf-8')).encode('ascii', 'ignore')
            if not word.lower() in lexicons_dict and word.isalnum():
                lexicons_dict[word.lower()] = id_counter
                id_counter += 1
            # sentence_temp = sentence_temp.replace('.', ' .')
            # sentence_temp = sentence_temp.replace('?', ' ?')
            # sentence_temp = sentence_temp.replace('-',' - ')
            # sentence_temp = sentence_temp.replace(',', ' ')
            # cleaned_evidence = sentence_temp.replace('[REF]', '')
            # cleaned_evidence = cleaned_evidence.replace('[REF', '')
            # cleaned_evidence = cleaned_evidence.replace('[STUDY]', '')
            # cleaned_evidence = cleaned_evidence.replace('[ANECDOTAL]', '')
            # cleaned_evidence = cleaned_evidence.replace('[EXPERT]', '')
            # cleaned_evidence = cleaned_evidence.replace('[STUDY, EXPERT]', '')
            # cleaned_evidence = cleaned_evidence.replace('[EXPERT]', '')
            # for word in cleaned_evidence.split():
            #     if not word.lower() in lexicons_dict and word.isalnum():
            #         lexicons_dict[word.lower()] = id_counter
            #         id_counter += 1

    print "\rCreating Dictionary ... Done!"

    print("\rFound %d unique lexicons" % len(lexicons_dict))

    return lexicons_dict, len(lexicons_dict)

def compile_data_evidence_motion(evidence_list):
    motion_evidence_map = {}

    fobj = csv.reader(open(evidence_list, "rb"), delimiter = '\t')
    for idx, line in enumerate(fobj):
        try:
            evidence_temp = unicodedata.normalize('NFKD', line[2].decode('utf-8')).encode('ascii', 'ignore')
            evidence_temp = evidence_temp.replace('.', ' .')
            evidence_temp = evidence_temp.replace('?', ' ?')
            evidence_temp = evidence_temp.replace('-',' - ')
            evidence_temp = evidence_temp.replace(',', ' ')
            cleaned_evidence = evidence_temp.replace('[REF]', '')
            cleaned_evidence = cleaned_evidence.replace('[REF', '')
            cleaned_evidence = cleaned_evidence.replace('[STUDY]', '')
            cleaned_evidence = cleaned_evidence.replace('[ANECDOTAL]', '')
            cleaned_evidence = cleaned_evidence.replace('[EXPERT]', '')
            cleaned_evidence = cleaned_evidence.replace('[STUDY, EXPERT]', '')
            cleaned_evidence = cleaned_evidence.replace('[EXPERT]', '')
            motion_evidence_map[cleaned_evidence] = line[0]
        except:
            continue

    return motion_evidence_map

def compile_data_claim_motion(claims_list):
    motion_claim_map = {}

    fobj = csv.reader(open(claims_list, "rb"), delimiter = '\t')
    for idx, line in enumerate(fobj):
        if idx == 0: continue
        try:
            sentence_temp = unicodedata.normalize('NFKD', line[2].decode('utf-8')).encode('ascii', 'ignore')
            sentence_temp = sentence_temp.replace('.', ' .')
            sentence_temp = sentence_temp.replace('?', ' ?')
            sentence_temp = sentence_temp.replace('-',' - ')
            sentence_temp = sentence_temp.replace(',', ' ')
            cleaned_claim = sentence_temp.replace('[REF]', '')
            motion_claim_map[cleaned_claim] = line[0]
        except:
            continue

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
                sentence = sentence.replace('?', ' ?')
                sentence = sentence.replace('-',' - ')
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

    motion_evidence_map, motion_claim_map, motion_article_map

    for key, value in motion_evidence_map.iteritems():
        encoded_motion = []
        for word in word_tokenize(value):
            if word.isalnum() or word == '?' or word == '.':
                try:
                    encoded_motion.append(lexicon_dictionary[word.lower()])
                except:
                    # print word
                    continue
            else :
                # print word
                continue

        encoded_sentence = []
        for word in word_tokenize(key):
            if word.isalnum() or word == '?' or word == '.':
                try:
                    encoded_sentence.append(lexicon_dictionary[word.lower()])
                except:
                    # print word
                    continue
            else :
                # print word
                continue

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
        for word in word_tokenize(value):
            if word.isalnum() or word == '?' or word == '.':
                try:
                    encoded_motion.append(lexicon_dictionary[word.lower()])
                except:
                    # print word
                    continue
            else :
                # print word
                continue

        encoded_sentence = []
        for word in word_tokenize(key):
            if word.isalnum() or word == '?' or word == '.':
                try:
                    encoded_sentence.append(lexicon_dictionary[word.lower()])
                except:
                    # print word
                    continue
            else :
                # print word
                continue

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
        for word in word_tokenize(value):
            if word.isalnum() or word == '?' or word == '.':
                try:
                    encoded_motion.append(lexicon_dictionary[word.lower()])
                except:
                    # print word
                    continue
            else :
                # print word
                continue

        encoded_sentence = []
        for word in word_tokenize(key):
            if word.isalnum() or word == '?' or word == '.':
                try:
                    encoded_sentence.append(lexicon_dictionary[word.lower()])
                except:
                    # print word
                    continue
            else :
                # print word
                continue

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
    histogram = "dataset/word_histogram.csv"
    processed_data_dir = join(task_dir, 'data_histo')
    train_data_dir = join(processed_data_dir, 'train')
    test_data_dir = join(processed_data_dir, 'test')

    if not exists(processed_data_dir):
        mkdir(processed_data_dir)
    if not exists(train_data_dir):
        mkdir(train_data_dir)
    if not exists(test_data_dir):
        mkdir(test_data_dir)

    articles_list = []

    if not exists(join(task_dir, 'data')):
        mkdir(join(task_dir, 'data'))

    for entryname in listdir(articles_dir):
        entry_path = join(articles_dir, entryname)
        if isfile(entry_path):
            articles_list.append(entry_path)

    if not isfile(join(processed_data_dir, 'lexicon-dict.pkl')):
        # lexicon_dictionary, lexicon_count = create_dictionary(articles=articles_list, motions=motions_list, claims=claims_list, evidence=evidence_list)
        lexicon_dictionary, lexicon_count = create_dictionary(histogram)
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
