import pickle
import time
import math
import numpy as np
import torch as th
import torch.nn.functional as F
from collections import defaultdict
import random
# import requests
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix

def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def save_pkl(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

def wechat_notify(title = 'title', text = 'test'):
    # TODO
    url = ''
    rsp=requests.get(url, {'text':title,'desp':text})

class dataHandler():
    '''
    for last session and similar session
    '''
    def __init__(self, data_path, batch_size = 128, n_friends = 10, friend_type='similar', device = 'cpu'):
        print(f'Loading dataset from {data_path}')
        stime = time.time()
        self.device = device
        self.batch_size = batch_size

        dataset = load_pkl(data_path)
        trainset = dataset['trainset']
        testset = dataset['testset']
        train_session_lengths = dataset['train_session_lengths']
        test_session_lengths = dataset['test_session_lengths']

        user_set = set(trainset.keys())
        self.num_users = len(trainset)
        self.n_friends = min(n_friends, self.num_users)
        self.friend_type = friend_type # 'similar','random'

        assert min(user_set) == 0
        assert (max(user_set)+1) == len(user_set)
        for user in testset.keys():
            assert user in user_set

        padding_item = -1
        self.num_session_train = 0
        self.num_session_test = 0
        self.user_item = defaultdict(set)
        self.user_item_negative = {}

        self.train_data = {}
        self.train_session_start_time = defaultdict(list)
        self.train_session_lengths = {}

        for user, session_list in trainset.items():
            assert len(session_list) >= 2
            for s in session_list:
                self.user_item[user].update(set([i[1] for i in s]))
            sessions = np.array(session_list)
            ordered_index = np.argsort(sessions[:,0,0])
            sess = th.from_numpy(sessions[:,:,1].astype(np.int64)[ordered_index]).to(self.device)
            self.train_data[user] = sess
            self.train_session_lengths[user] = np.array(train_session_lengths[user])[ordered_index]
            padding_item = max(padding_item, sessions[:,:,1].max())
            self.num_session_train += len(session_list)

        self.test_data = {}
        self.test_session_lengths = {}
        for user, session_list in testset.items():
            assert len(session_list) >= 1
            for s in session_list:
                self.user_item[user].update(set([i[1] for i in s]))
            sessions = np.array(session_list)
            ordered_index = np.argsort(sessions[:,0,0])
            sess = th.from_numpy(sessions[:,:,1].astype(np.int64)[ordered_index]).to(self.device)
            self.test_data[user] = sess
            self.test_session_lengths[user] = np.array(test_session_lengths[user])[ordered_index]
            self.num_session_test += len(session_list)
        self.padding_item = int(padding_item) # max index of items is the padding item
        self.num_items = int(padding_item) 

        # for user, items in self.user_item.items():
        #     negatives = [i for i in range(self.num_items) if i not in items]
        #     self.user_item_negative[user] = np.array(negatives)

        if n_friends > 0 and self.friend_type == 'similar':
            self.user_similarity = self.get_similar_users()
        
        print(f'Dataset loaded in {(time.time()-stime):.1f}s')

    def get_similar_users(self):
        row = []
        col = []
        for usr, itms in self.user_item.items():
            col.extend(list(itms))
            row.extend([usr]*len(itms))
        row = np.array(row)
        col = np.array(col)
        idxs = col != self.padding_item
        col = col[idxs]
        row = row[idxs] #! user id start from 0 to N-1
        feature_mtx = coo_matrix(([1]*len(row),(row,col)),shape=(self.num_users, self.num_items))
        similarity = cosine_similarity(feature_mtx)
        return similarity.argsort()[:,-(self.n_friends+1):]  

    def reset_batch(self, dataset, start_index = 1):
        self.num_remain_sessions = np.zeros(self.num_users, int)
        self.index_cur_session = np.ones(self.num_users, int) * start_index
        for user, session_list in dataset.items():
            self.num_remain_sessions[user] = len(session_list) - start_index
        assert self.num_remain_sessions.min() >= 0
    
    def reset_train_batch(self):
        self.reset_batch(self.train_data, start_index = 1)

    def reset_test_batch(self):
        self.reset_batch(self.test_data, start_index = 0)

    # @timeit
    def get_next_batch(self, dataset, dataset_session_lengths, training = True):
        # select users for the batch
        if (self.num_remain_sessions>0).sum() >= self.batch_size: # 有剩余session的用户数量大于batch size，需要从中选择哪些用户
            batch_users = np.argsort(self.num_remain_sessions)[-self.batch_size:]
        else: # 否则所有还有剩余session的用户组成batch，数量小于batch size
            batch_users = np.where(self.num_remain_sessions > 0)[0]
        
        if len(batch_users) == 0:
            # end of the epoch
            return batch_users, None, None, None, None

        cur_sess = [] # current sessions
        cur_sess_len = []
        # itm_negative = []
        hist_sess = [] # history sessions for each user
        friend_sess = [] # friends' sessions for each user

        for user in batch_users:
            cur_sess.append(dataset[user][self.index_cur_session[user],:])
            cur_sess_len.append(dataset_session_lengths[user][self.index_cur_session[user]])
            # idx_negs = random.choices(range(len(self.user_item_negative[user])), k=n_negative)
            # itm_negative.append(th.LongTensor(self.user_item_negative[user][idx_negs]).to(self.device))

            if training:
                prev_sess = self.train_data[user][:self.index_cur_session[user],:]
            else:
                if self.index_cur_session[user] > 0:
                    prev_sess = th.cat([self.train_data[user], self.test_data[user][:self.index_cur_session[user],:]], dim=0) # !需要保证测试集的session都在训练集之后
                else:
                    prev_sess = self.train_data[user]
            hist_sess.append(prev_sess)

            # friend sessions
            if self.n_friends > 0:
                friend_sess_item_list = []
                if self.friend_type == 'similar':
                    for frd in self.user_similarity[user]:
                        if frd != user:
                            friend_sess_item_list.append(self.train_data[frd])
                elif self.friend_type == 'random':
                    for idx, frd in enumerate(random.sample(range(self.num_users), k=(self.n_friends+1))):
                        if frd != user:
                            friend_sess_item_list.append(self.train_data[frd])
                        if idx == self.n_friends:
                            break
                else:
                    raise NotImplementedError('friend_type not supported')
                sess_itm_data = th.cat((friend_sess_item_list), dim=0)
                friend_sess.append(sess_itm_data)

            self.index_cur_session[user] += 1
            self.num_remain_sessions[user] -= 1
        return batch_users, th.cat(cur_sess).view(len(batch_users),-1), np.array(cur_sess_len), hist_sess, friend_sess

    def get_next_train_batch(self):
        return self.get_next_batch(self.train_data, self.train_session_lengths, training = True)

    def get_next_test_batch(self):
        return self.get_next_batch(self.test_data, self.test_session_lengths, training = False)

    def get_num_remain_batches(self):
        return math.ceil(self.num_remain_sessions.sum()/self.batch_size)

class Tester:
    def __init__(self, session_length = 5, k_list=[5, 10, 20]):
        self.k_list = k_list
        self.session_length = session_length
        self.n_decimals = 4
        self.initialize()

    def initialize(self):
        self.i_count = np.zeros(self.session_length)#[0]*self.session_length
        self.recall = np.zeros((self.session_length, len(self.k_list)))#[[0]*len(self.k) for i in range(self.session_length)]
        self.mrr = np.zeros((self.session_length, len(self.k_list)))#[[0]*len(self.k) for i in range(self.session_length)]
        self.ndcg = np.zeros((self.session_length, len(self.k_list)))

    def get_rank(self, target, predictions):
        for i in range(len(predictions)):
            if target == predictions[i]:
                return i+1

        raise Exception("could not find target in sequence")

    def evaluate_sequence(self, predicted_sequence, target_sequence, seq_len):
        for i in range(min(self.session_length, seq_len)):
            target_item = target_sequence[i]
            k_predictions = predicted_sequence[i]

            for j in range(len(self.k_list)):
                k = self.k_list[j]
                if target_item in k_predictions[:k]:
                    self.recall[i][j] += 1
                    rank = self.get_rank(target_item, k_predictions[:k])
                    self.mrr[i][j] += 1.0/rank
                    self.ndcg[i][j] += 1 / math.log(rank + 1, 2)
            self.i_count[i] += 1


    def evaluate_batch(self, predictions, targets, sequence_lengths):
        for batch_index in range(len(predictions)):
            predicted_sequence = predictions[batch_index]
            target_sequence = targets[batch_index]
            self.evaluate_sequence(predicted_sequence, target_sequence, sequence_lengths[batch_index])
    
    # def evaluate_sequence2(self, predicted_sequence, target_sequence, seq_len):
    #     i = seq_len-1
    #     target_item = target_sequence
    #     k_predictions = predicted_sequence

    #     for j in range(len(self.k_list)):
    #         k = self.k_list[j]
    #         if target_item in k_predictions[:k]:
    #             self.recall[i][j] += 1
    #             inv_rank = 1.0/self.get_rank(target_item, k_predictions[:k])
    #             self.mrr[i][j] += inv_rank

    #     self.i_count[i] += 1

    # def evaluate_batch2(self, predictions, targets, sequence_lengths):
    #     for batch_index in range(len(predictions)):
    #         predicted_sequence = predictions[batch_index]
    #         target_sequence = targets[batch_index]
    #         self.evaluate_sequence2(predicted_sequence, target_sequence, sequence_lengths[batch_index])
    
    # def evaluate_batch3(self, output, sequence_lengths, batch_users):
    #     for sess, slen in zip(output, sequence_lengths):
    #         eval_len = min(slen, self.session_length)
    #         sess_ = sess[:eval_len]
    #         for idxk, k in enumerate(self.k_list):
    #             recall = np.append((sess_ <= k).astype(int), [0]*(self.session_length-eval_len))
    #             inv_rank = 1/(sess_+1)
    #             self.recall[:,idxk] += recall
    #             self.mrr[:,idxk] += np.append(inv_rank, [0]*(self.session_length-eval_len))
                
    #         count = np.array([1] * eval_len + [0] * (self.session_length-eval_len))
    #         self.i_count += count

    # def format_score_string(self, score_type, score):
    #     tabs = '\t'
    #     return '\t'+score_type+tabs+score+'\n'

    def get_stats(self):
        score_message = "Position\tR@5   \tMRR@5 \tNDCG@5\tR@10   \tMRR@10\tNDCG@10\tR@20  \tMRR@20\tNDCG@20\n"
        current_recall = np.zeros(len(self.k_list))
        current_mrr = np.zeros(len(self.k_list))
        current_ndcg = np.zeros(len(self.k_list))
        current_count = 0
        recall_k = np.zeros(len(self.k_list))
        for i in range(self.session_length):
            score_message += "\ni<="+str(i+2)+"    \t"
            current_count += self.i_count[i]
            for j in range(len(self.k_list)):
                current_recall[j] += self.recall[i][j]
                current_mrr[j] += self.mrr[i][j]
                current_ndcg[j] += self.ndcg[i][j]

                r = current_recall[j]/current_count
                m = current_mrr[j]/current_count
                n = current_ndcg[j]/current_count
                
                score_message += str(round(r, self.n_decimals))+'\t'
                score_message += str(round(m, self.n_decimals))+'\t'
                score_message += str(round(n, self.n_decimals))+'\t'

                recall_k[j] = r

        recall5 = recall_k[0]
        recall20 = recall_k[2]

        return score_message, recall5, recall20

    # def get_stats_user(self):
    #     return self.u_recall/(np.expand_dims(self.u_count,1)+1e-10)

    # def get_stats2(self):
    #     score_message = "Recall@5\tMRR@5\tRecall@10\tMRR@10\tRecall@20\tMRR@20\n"
    #     current_recall = np.zeros(len(self.k_list))
    #     current_mrr = np.zeros(len(self.k_list))
    #     current_count = 0
    #     recall_k = np.zeros(len(self.k_list))
    #     for i in range(self.session_length):
    #         score_message += "\ni<="+str(i)+"\t"
    #         current_count += self.i_count[i]
    #         for j in range(len(self.k_list)):
    #             current_recall[j] += self.recall[i][j]
    #             current_mrr[j] += self.mrr[i][j]
    #             k = self.k_list[j]

    #             r = current_recall[j]/current_count
    #             m = current_mrr[j]/current_count
                
    #             score_message += str(round(r, self.n_decimals))+'\t'
    #             score_message += str(round(m, self.n_decimals))+'\t'

    #             recall_k[j] = r

    #     recall5 = recall_k[0]
    #     recall20 = recall_k[2]

    #     return score_message, recall5, recall20

    # def get_stats_and_reset(self):
    #     message = self.get_stats()
    #     self.initialize()
    #     return message
