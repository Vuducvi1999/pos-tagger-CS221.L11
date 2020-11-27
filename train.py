import pickle
import gzip
import time
import random
import nltk
import numpy as np
import pandas as pd

# nltk.download('treebank')
# nltk.download('universal_tagset')


nltk_data = nltk.corpus.treebank.tagged_sents(tagset='universal')

# train_set, test_set = nltk_data[:int(
#     len(nltk_data)*0.8)], nltk_data[int(len(nltk_data)*0.8):]

train_set = nltk_data

# train_set = [
#     [('con', 'UN'), ('mèo', 'NN'), ('trèo', 'VB'),
#      ('cây', 'UN'), ('cau', 'NN')],
#     [('con', 'UN'), ('chuột', 'NN'), ('trèo', 'VB'),
#      ('cây', 'UN'), ('cau', 'NN')],
#     [('con', 'UN'), ('chuột', 'NN'), ('hỏi', 'VB'),
#      ('con', 'UN'), ('mèo', 'NN')],
#     [('con', 'UN'), ('trèo', 'VB'), ('là', 'VB'),
#      ('con', 'UN'), ('nào', 'PRP')]
# ]


train_tagged_words = [tup for sent in train_set for tup in sent]
# test_tagged_words = [tup for sent in test_set for tup in sent]


# tất cả các loại tag có trong train_data
tags = list({tag for word, tag in train_tagged_words})
# all_tags = [tag for word, tag in train_tagged_words]
all_words = list({word for word, tag in train_tagged_words})


# tính toán ma trận thể hiện Emission Probability
def word_given_tag(word, tag, train_tagged_words=train_tagged_words):
    # tìm tất cả các tag khả thi
    tag_list = [pair for pair in train_tagged_words if pair[1] == tag]
    count_tag = len(tag_list) + len(all_words)
    # tìm tất cả các word có tag khả thi
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]
    count_w_given_tag = len(w_given_tag_list) + 1
    return (count_w_given_tag, count_tag)


# tính toán ma trận chuyển đổi Transition Probability
def t2_given_t1(t2, t1):
    if t1 == '<start>':
        count_t1 = len(train_set) + len(tags)
        count_t2_t1 = 1
        for i in train_set:
            if i[0][1] == t2:
                count_t2_t1 += 1
        return (count_t2_t1, count_t1)

    # số lượng tag1 (không kể tag cuối)
    count_t1 = len([j for i in train_set for index, j in enumerate(
        i) if j[1] == t1 and index != len(i)-1]) + len(tags)
    # số lượng tag2 liền kề tag1
    count_t2_t1 = 1
    for i in train_set:
        for j in range(len(i)-1):
            if i[j][1] == t1 and i[j+1][1] == t2:
                count_t2_t1 += 1
    return (count_t2_t1, count_t1)


# Tính toán ma trận chuyển đổi
tags_matrix = np.zeros((len(tags)+1, len(tags)), dtype='float32')
for i, t1 in enumerate(tags):
    for j, t2 in enumerate(tags):
        tags_matrix[i, j] = t2_given_t1(
            t2, t1)[0]/t2_given_t1(t2, t1)[1]
for i, t in enumerate(tags):
    tags_matrix[len(tags), i] = t2_given_t1(t, '<start>')[
        0]/t2_given_t1(t, '<start>')[1]

new_tags = tags.copy()
new_tags.append('<start>')
# Ma trận chuyển đổi Transition Probability
tags_df = pd.DataFrame(tags_matrix, columns=list(
    tags), index=new_tags)


# Tính toán ma trận thể hiện
words_matrix = np.zeros((len(tags), len(all_words)), dtype='float32')
for i, t1 in enumerate(tags):
    for j, t2 in enumerate(all_words):
        words_matrix[i, j] = word_given_tag(
            t2, t1)[0]/word_given_tag(t2, t1)[1]
# Ma trận thể hiện Emission Probability
words_df = pd.DataFrame(words_matrix, columns=all_words, index=tags)


# nén ma trận chuyển đổi
model_transition = 'Transition_probability.data'
fp = gzip.open(model_transition, 'wb')
pickle.dump(tags_df, fp)
fp.close()


# nén ma trận thể hiện
model_emission = 'Emission_probability.data'
fp = gzip.open(model_emission, 'wb')
pickle.dump(words_df, fp)
fp.close()


# nén tags
model_emission = 'Tags.data'
fp = gzip.open(model_emission, 'wb')
pickle.dump(tags, fp)
fp.close()


# giải nén ma trận chuyển đổi
# fp = gzip.open(model_transition, 'rb')
# model = pickle.load(fp)
# fp.close()
# print(model)


# giải nén ma trận thể hiện
# fp = gzip.open(model_emission, 'rb')
# model = pickle.load(fp)
# fp.close()
# print(model)
