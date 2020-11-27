import nltk
import numpy as np
import pandas as pd
import gzip
import pickle

# [('Pierre', 'NOUN'), ('Vinken', 'NOUN'), (',', '.'), ('61', 'NUM'),
# ('years', 'NOUN'), ('old', 'ADJ'), (',', '.'), ('will', 'VERB'), ('join', 'VERB'),
# ('the', 'DET'), ('board', 'NOUN'), ('as', 'ADP'), ('a', 'DET'), ('nonexecutive', 'ADJ'),
# ('director', 'NOUN'), ('Nov.', 'NOUN'), ('29', 'NUM'), ('.', '.')]


model_transition = 'Transition_probability.data'
model_emission = 'Emission_probability.data'
model_tags = 'Tags.data'

# giải nén ma trận chuyển đổi
fp = gzip.open(model_transition, 'rb')
tags_df = pickle.load(fp)
fp.close()

# giải nén ma trận thể hiện
fp = gzip.open(model_emission, 'rb')
words_df = pickle.load(fp)
fp.close()

# giải nén tags
fp = gzip.open(model_tags, 'rb')
tags = pickle.load(fp)
fp.close()

# Tất cả các words
all_words = words_df.columns.tolist()


# thuật toán viterbi
def Viterbi(words):
    matrix_state = np.zeros((len(words), len(tags))).tolist()
    for key, word in enumerate(words):
        for index, tag in enumerate(tags):
            if word not in all_words:
                emission_p = 1/len(all_words)
            else:
                # xác suất thể hiện (nhờ ma trận Emission Probability) của word với tag hiện tại
                emission_p = words_df.loc[tag, words[key]]

            if key == 0:
                # xác suất chuyển đổi (nhờ ma trận Transition Probability)
                transition_p = tags_df.loc['<start>', tag]
                state_probability = emission_p * transition_p
                matrix_state[key][index] = (
                    state_probability * 10000000, 'Start')
            else:
                # array các xác suất tag hiện tại của word
                p = []
                for i, t in enumerate(tags):
                    temp = (matrix_state[key-1][i][0] * 10000000 *
                            emission_p * tags_df.loc[t, tag], t)
                    p.append(temp)
                a, b = zip(*p)
                maxp = max(a)
                for i in p:
                    if i[0] == maxp:
                        matrix_state[key][index] = i
                        break
    return matrix_state


def Tagger(text):
    text_test = text
    # list các tag kết quả
    state = []
    # Tìm ma trận xác suất
    tags_predict = Viterbi(text_test.split())

    temp2 = text_test.split()
    temp2.reverse()
    temp1 = tags_predict.copy()
    temp1.reverse()
    # ma trận xác suất sử dụng pandas
    tags_predict = pd.DataFrame(temp1, columns=list(tags), index=temp2)
    # print(tags_predict)
    file_path = 'matrix.txt'
    f = open(file_path, 'w', encoding='utf-8')
    f.write(tags_predict.to_string())
    f.close()

    columns_of_tags_predict = tags_predict.columns.tolist()
    arr_of_tags_predict = tags_predict.values.tolist()

    # Tìm list các tag kết quả
    for index, i in enumerate(arr_of_tags_predict):
        if index == 0:
            a, b = zip(*i)
            maxp = max(a)
            state.append(columns_of_tags_predict[a.index(maxp)])
        else:
            pre_tag = state[-1]
            current_index = columns_of_tags_predict.index(pre_tag)
            state.append(arr_of_tags_predict[index-1][current_index][1])

    state.reverse()
    return state


# modified Viterbi to include rule based tagger in it
def Viterbi_rule_based(words):
    state = []
    T = tags
    probal = []
    words = words.split()
    for key, word in enumerate(words):
        # initialise list of probability column for a given observation
        p = []
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            if word not in all_words:
                emission_p = 1/len(all_words)
            else:
                # xác suất thể hiện (nhờ ma trận Emission Probability) của word với tag hiện tại
                emission_p = words_df.loc[tag, words[key]]

            state_probability = emission_p * transition_p
            p.append(state_probability)

        pmax = max(p)
        probal.append(pmax)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)]
        state.append(state_max)
    return state


# print(Viterbi_rule_based(
#     'Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .'))

print(Tagger('As seen above , using the Viterbi algorithm along with rules can yield us better results'))
