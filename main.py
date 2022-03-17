import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pathlib import Path as path
from utils import dataHandler, Tester#, notify
import argparse
import os
import numpy as np
import dgl
# import dgl.function as fn
import time
import datetime
# from tqdm import tqdm
import progressbar

'''
RecSys的INSERT模型实现
'''

parser = argparse.ArgumentParser()

# hyperparameters
parser.add_argument("--num_iter", default=500, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--dim_hidden", default=50, type=int)
parser.add_argument("--dropout", default=0.2, type=float)
parser.add_argument("--l2", default=0, type=float)

# data
parser.add_argument("--data", default='delicious')
# history sessions
parser.add_argument("--ctype", default='all', help='self|hist|all')
# friend: 
parser.add_argument("--n_friends", default=10, type=int,help='the number of selected users, works when ctype=neig')
parser.add_argument("--friendtype", default='similar', help='similar|random')
# parser.add_argument("--betasim", default=0.1, type=float, help='weight of sim2')
parser.add_argument("--encoder_type", default='rnn', help='rnn|avg')
parser.add_argument("--avg_len", default=2, type=int, help='>=1, works when encoder_type=avg')

# training
parser.add_argument("--no-cuda", dest='nocuda', action='store_true')
parser.add_argument("--gpu", default=(torch.cuda.device_count()-1), type=int, help='for multiple gpus, default the last one')
parser.add_argument("--c", default='test', help='comments')
parser.add_argument("--load", default='', help='model file to load parameters')
parser.add_argument("--save", dest='save', action='store_true')
parser.add_argument("--eval", dest='eval', action='store_true')
parser.add_argument("--eval_every", default=2, type=int, help='evaluation frequency')
parser.add_argument("--patience", default=5, type=int, help='early stopping')
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--notify", dest='notify', action='store_true')

parser.set_defaults(nocuda = False)
parser.set_defaults(save = False)
parser.set_defaults(eval = False) # no traning parse
parser.set_defaults(notify = False) # notify via wechat
parser.set_defaults(valset = True) # if use validate set
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and not args.nocuda else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True

# just a test
sim_all = False # 只使用item embedding平均来计算session similarity
if sim_all:
    args.encoder_type='avg_all'
ses_emb_all = False # 召回结果使用item embedding平均

# model;
class INSERT(nn.Module):
    def __init__(self, hidden_size, num_users, num_items, n_friends=1,\
                 encoder_type = 'rnn', avg_len = 2,  dropout = 0.2, ctype = 'all'):
        super(INSERT, self).__init__()
        self.hidden_size = hidden_size
        self.padding_item = num_items # id of padding item is the number of items
        self.num_users = num_users
        self.ctype = ctype
        self.dropout = dropout
        self.avg_len = avg_len
        self.n_friends = n_friends
        self.encoder_type = encoder_type
        self.max_sess_len = 20 # max session length in dataset

        if self.ctype == 'self': # only current session
            self.n_channel = 1
        elif n_friends > 0:
            self.n_channel = 3
        else:
            self.n_channel = 2

        self.v_embedding = nn.Embedding((num_items+1), hidden_size, padding_idx=self.padding_item)
        self.u_embedding = nn.Embedding(num_users, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True)
        self.out2 = Parameter(torch.Tensor(hidden_size*self.n_channel, hidden_size))
        self.out1 = nn.Linear(hidden_size*self.n_channel, num_items)

        self.attn_sess = nn.Linear(hidden_size, 1)

        if self.avg_len > 0:
            _avg_mtx = torch.ones(self.max_sess_len,self.max_sess_len) # should >= the max length of sessions
            self.avg_mtx = (torch.triu(_avg_mtx) - torch.triu(_avg_mtx, diagonal = self.avg_len)).to(device)
            self.avg_mtx = self.avg_mtx / self.avg_mtx.sum(0).unsqueeze(0)

        self.__init_params()

    def __init_params(self):
        torch.nn.init.uniform_(self.v_embedding.weight[:-1], -1, 1) # 初始化需要避开padding item
        torch.nn.init.xavier_uniform_(self.out2)
        torch.nn.init.xavier_uniform_(self.out1.weight)

    def forward(self, users, cur_sess, hist_sess, frds_sess, cur_sess_len):
        # current session
        # if self.encoder_type == 'rnn':
        hs, _ = self.rnn_encoder(cur_sess, cur_sess_len)
        # elif self.encoder_type == 'avg':
        #     hs, _ = self.avg_encoder(cur_sess, cur_sess_len, all=True)

        if self.ctype == 'self':
            output = hs
        else:
            query = hs#self.avg_encoder(cur_sess, cur_sess_len)[0]
            hist_sess_graph, frd_sess_graph = self.build_graph(users, hist_sess, frds_sess)
            hu, hf = self.hist_emb(query, hist_sess_graph, frd_sess_graph)

            # TODO 去掉用户历史
            hu = torch.zeros_like(hs).to(device)

            output = torch.cat([hs, hu, hf], dim=-1)

        
        # output = torch.tanh(output)
        output = F.dropout(output, p=self.dropout)
        # bilinear
        if args.data=='delicious':
            output = torch.mm(output.view(-1, self.hidden_size*self.n_channel), self.out2)
            output = torch.mm(output.view(-1, self.hidden_size), self.v_embedding.weight[:-1].T)
        # mlp
        else:
            output = self.out1(output)
        
        return output.view(hs.size(0),hs.size(1),-1)

    def rnn_encoder(self, session, sess_len):
        h = self.initHidden(batch_size=len(session))
        hs = F.dropout(self.v_embedding(session), p=self.dropout)

        if isinstance(sess_len, np.ndarray):
            sess_len = torch.Tensor(sess_len)

        hs = torch.nn.utils.rnn.pack_padded_sequence(hs, sess_len.cpu(), batch_first=True, enforce_sorted=False)
        hs, h = self.gru(hs, h)
        hs, _ = torch.nn.utils.rnn.pad_packed_sequence(hs, batch_first=True)
        return hs, h

    def avg_encoder(self, session, sess_len, all=False):
        max_len = sess_len.max()
        hs = F.dropout(self.v_embedding(session[:,:max_len]), p=self.dropout)

        hs = hs.permute([0,2,1]).reshape(-1,max_len)
        if not all:
            hs = torch.mm(hs, self.avg_mtx[:max_len,:max_len])
        else:
            hs = torch.mm(hs, self.avg_mtx_all[:max_len,:max_len])
        hs = hs.reshape(-1,self.hidden_size,max_len).permute([0,2,1])
        h = []
        # t = (ll[:,:max_len]-hs)
        return hs, h
    
    def avg_all(self, session, sess_len):
        max_len = sess_len.max() # 做对比实验，待查询的session 使用item embedding的平均作为session表示
        hs = F.dropout(self.v_embedding(session[:,:max_len]), p=self.dropout)
        hs = hs.sum(1)
        hs = hs/max_len.view(-1,1)
        return hs.unsqueeze(1), hs

    def build_graph(self, users, history_session, friend_session):
        # sessions from own history
        hisory_session_graph = []
        for u, hs in zip(users, history_session):
            # if self.ctype == 'last':
            #     session_link_s2u = [(0,0)]
            #     session_items = hs[-1].unsqueeze(0)
            # else:
            session_link_s2u = [(idx_sess, 0) for idx_sess in range(len(hs))]
            session_items = hs
            session_graph = dgl.heterograph({('session','sess2user','user'):session_link_s2u})
            session_graph.nodes['session'].data['item_list'] = session_items
            session_graph.nodes['user'].data['user_emb'] = self.u_embedding(torch.LongTensor([u]).to(device))
            hisory_session_graph.append(session_graph)
        hisory_session_graph = dgl.batch_hetero(hisory_session_graph)

        # sessions from other users
        friend_session_graph = []
        if self.n_friends > 0:
            for u, fs in zip(users, friend_session):
                session_link_s2u = [(idx_sess, 0) for idx_sess in range(len(fs))]
                session_items = fs
                session_graph = dgl.heterograph({('session','sess2user','user'):session_link_s2u})
                session_graph.nodes['session'].data['item_list'] = session_items
                # if self.ctype == 'last':
                #     session_items_last = torch.cat([(session_items[0]).unsqueeze(0),session_items[:-1]])
                #     session_graph.nodes['session'].data['item_list_last'] = session_items_last
                session_graph.nodes['user'].data['user_emb'] = self.u_embedding(torch.LongTensor([u]).to(device))
                friend_session_graph.append(session_graph)
            friend_session_graph = dgl.batch_hetero(friend_session_graph)
        return hisory_session_graph, friend_session_graph

    def sess_encoder(self, nodes):
        # item id list of the sessions
        item_list = nodes.data['item_list']
        # the lengths of the sessions
        # sess_len = np.ones(item_list.size()[0])*item_list.size()[1]
        # idx_to_updt = ((item_list==self.padding_item).sum(1)>0).cpu().numpy()
        # sess_len[idx_to_updt] = (torch.argmax((item_list==self.padding_item)*self.reversed_idx,1)[idx_to_updt]).cpu().numpy()
        sess_len = (item_list != self.padding_item).sum(1)
        # feed to a rnn
        if self.encoder_type == 'rnn':
            hs, h = self.rnn_encoder(item_list, sess_len)
            h = h[0]
        elif self.encoder_type == 'avg':
            hs, h = self.avg_encoder(item_list, sess_len)
        else: # baseline，待查询session使用item平均作为表示，encoder_type不为rnn和avg时启用
            hs, h = self.avg_all(item_list, sess_len) # 计算所有item的平均
        return {'encoder_output': hs, 'encoder_output_all': h}
        # hs = self.v_embedding(item_list)
        # hs = F.dropout(hs, p=self.dropout)
        # h = self.initHidden(batch_size=nodes.batch_size())
        #
        # hs = torch.nn.utils.rnn.pack_padded_sequence(hs, sess_len, batch_first=True, enforce_sorted = False)
        # hs, h = self.gru(hs, h)
        # hs, _ = torch.nn.utils.rnn.pad_packed_sequence(hs, batch_first=True)

        # if 'item_list_last' in nodes.data:
        #     last_itemlist = nodes.data['item_list_last']
        #     # the lengths of the sessions
        #     sess_len = np.ones(last_itemlist.size()[0])*last_itemlist.size()[1]
        #     idx_to_updt = ((last_itemlist==self.padding_item).sum(1)>0).cpu().numpy()
        #     sess_len[idx_to_updt] = (torch.argmax((last_itemlist==self.padding_item)*self.reversed_idx,1)[idx_to_updt]).cpu().numpy()
        #
        #     # feed to a rnn
        #     hs1 = self.v_embedding(last_itemlist)
        #     hs1 = F.dropout(hs1, p=self.dropout)
        #     h = self.initHidden(batch_size=nodes.batch_size())
        #
        #     hs1 = torch.nn.utils.rnn.pack_padded_sequence(hs1, sess_len, batch_first=True, enforce_sorted = False)
        #     hs1, h = self.gru(hs1, h)
        #     hs1, _ = torch.nn.utils.rnn.pad_packed_sequence(hs1, batch_first=True)

         #, [0]TODO avg情况还没处理 若存在item_list_last则last_hdn保存的是上个session的last hd，否则是当前session

    def sess_embedding_attn(self, edges):
        # key/value embedding
        target_itemlist = edges.src['item_list']
        sess_len = (target_itemlist!=self.padding_item).sum(2)
        max_sess_len = sess_len.max().item()
        sess_itms = F.dropout(self.v_embedding(target_itemlist[:,:,:max_sess_len]), p=self.dropout)
        if not ses_emb_all:
            user_emb = edges.dst['user_emb'].unsqueeze(2)

            ## calculate similarity
            weight = (user_emb * sess_itms).sum(-1).softmax(2)
            emb = (sess_itms * weight.unsqueeze(-1)).sum(2)
        # session embedding: average of item embeddings
        else:
            hs = sess_itms.sum(2)
            emb = hs/sess_len.unsqueeze(-1)
        # emb2 = h

        return {'sess_emb': emb}

    def sess_similarity(self, edges):
        topk = 2 # 对每个query保持的最相似session个数，不同query的结果会合并，所以每个user最终保留的session个数会大于2
        result = {}
        if 'step_qry' in edges.dst:
            step_qry = edges.dst['step_qry']
            subsess_emb = edges.src['encoder_output']
            # test no subsession
            if sim_all:
                sess_emb = edges.src['encoder_output_all']
                subsess_emb = sess_emb.unsqueeze(2)
            # end test
            sim1, idx_itms = (step_qry.unsqueeze(3) * subsess_emb.unsqueeze(2)).sum(-1).max(3)
            sim1 = torch.softmax(sim1, dim=1)
            result['sim1'] = sim1
            # result['sim_emb'] = torch.gather(edges.src['encoder_output'],2,idx_itms.unsqueeze(-1).expand(idx_itms.size(0),idx_itms.size(1),idx_itms.size(2),self.hidden_size))
            # filtering links
            if step_qry.size(1) > topk:
                preserve1 = torch.zeros(step_qry.size(0),step_qry.size(1)).bool().to(device)
                for i,l in enumerate(sim1.argsort(dim=1,descending=True)[:,:topk]):
                    preserve1[i,l.unique()] = True
                result['prev'] = preserve1
            else:
                result['prev'] = torch.ones(step_qry.size(0),step_qry.size(1)).bool().to(device)

        # if 'sim_itm' in edges.dst: # TODO,如何利用sim_itm计算相似度
        #     sim_itm = edges.dst['sim_itm']
        #     step_rnn = edges.src['encoder_output']
        #     sim2, _ = (sim_itm.unsqueeze(3) * step_rnn.unsqueeze(2)).sum(-1).max(3)
        #     sim2 = torch.softmax(sim2, dim=1)
        #     result['sim2'] = sim2
        #     # for filtering links
        #     if sim_itm.size(1) > topk:
        #         preserve2 = torch.zeros(sim_itm.size(0),sim_itm.size(1)).bool().to(device)
        #         for i,l in enumerate(sim2.argsort(dim=1,descending=True)[:,:topk]):
        #             preserve2[i,l.unique()] = True
        #         if 'prev' in result:
        #             result['prev'] = (result['prev'] + preserve2)
        #         else:
        #             result['prev'] = preserve2
        #     else:
        #         result['prev'] = torch.ones(sim_itm.size(0),sim_itm.size(1)).bool().to(device)

        return result

    def user_sess_reduce_func(self, node):
        sim = 0
        if 'sim1' in node.mailbox:
            sim1 = node.mailbox['sim1']
            sim = sim + sim1.softmax(1)

        # if 'sim2' in node.mailbox: #TODO 暂时不考虑两个similarity
        #     sim2 = node.mailbox['sim2']
        #     sim = sim + sim2.softmax(1) * args.betasim

        sim = sim.softmax(1).unsqueeze(3)

        sess_embs = node.mailbox['sess_emb']
        hist_emb = (sim * sess_embs.unsqueeze(2)).sum(1)
        
        result = {
                'hist_emb': hist_emb
                }

        # if 'sim2' not in node.mailbox: # TODO for history reduce
        #     sim_itm = node.mailbox['sim_itm'] # for query from other users
        #     sim_itm = (sim * sim_itm).sum(1)
        #     result['sim_itm'] = sim_itm # batch*neigs*nquery*hidden
        return result

    def user_sess_copy_func(self, edge):
        result = {}
        for k in ['sim1','sim2','sess_emb','sim_emb']:
            if k in edge.data:
                result[k] = edge.data[k]
        return result

    def hist_emb(self, query, hist_sess, frd_sess):

        ## 1. feed query embeddings to history sessions
        hist_sess.nodes['user'].data['step_qry'] = F.dropout(query,p=self.dropout)
        ## 2. calculating session similarity and filtering dissimilar links
        hist_sess.apply_nodes(self.sess_encoder, ntype='session')

        hist_sess.group_apply_edges('dst',self.sess_similarity, etype='sess2user')
        filter_hist_sess = hist_sess.filter_edges(lambda edges: edges.data['prev'], etype='sess2user')
        filtered_hist_sess = hist_sess.edge_subgraph({'sess2user':filter_hist_sess})
        ## 3. session embedding
        filtered_hist_sess.group_apply_edges('dst',self.sess_embedding_attn, etype='sess2user')
        ## 4. user history embedding
        filtered_hist_sess.send_and_recv(filtered_hist_sess['sess2user'].edges(),\
            self.user_sess_copy_func, self.user_sess_reduce_func, etype='sess2user')
        hist_result = filtered_hist_sess.nodes['user'].data.pop('hist_emb')
        # sim_itm = filtered_hist_sess.nodes['user'].data.pop('sim_itm')
        
        ## 3. session from other users
        if self.n_friends > 0:
            frd_sess.nodes['user'].data['step_qry'] = F.dropout(query,p=self.dropout)
            # frd_sess.nodes['user'].data['sim_itm'] = F.dropout(sim_itm,p=self.dropout)
            # similarity
            frd_sess.apply_nodes(self.sess_encoder, ntype='session')
            frd_sess.group_apply_edges('dst',self.sess_similarity, etype='sess2user')
            filter_frd_links = frd_sess.filter_edges(lambda edges: edges.data['prev'], etype='sess2user')
            filtered_frd_sess = frd_sess.edge_subgraph({'sess2user':filter_frd_links})

            filtered_frd_sess.group_apply_edges('dst',self.sess_embedding_attn, etype='sess2user')
            # aggregating
            filtered_frd_sess.send_and_recv(filtered_frd_sess['sess2user'].edges(),\
                self.user_sess_copy_func, self.user_sess_reduce_func, etype='sess2user')
            friend_result = filtered_frd_sess.nodes['user'].data.pop('hist_emb')
        else:
            friend_result = torch.empty(0).to(device)
        return hist_result, friend_result
    
    def loss(self, output, cur_sess_len):
        target = self.get_target(cur_sess, cur_sess_len)
        return F.cross_entropy(output.view(-1,self.padding_item), target.flatten(), ignore_index = self.padding_item)
        
    def predict(self, users, cur_sess, hist_sess, frds_sess, cur_sess_len, topk = 50):
        with torch.no_grad():
            output = self.forward(users, cur_sess, hist_sess, frds_sess, cur_sess_len)
            return torch.topk(output, topk, dim=2)

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
    
    def get_target(self, cur_sess, cur_sess_len):
        return cur_sess[:,1:cur_sess_len.max()+1]

print(f'Algorithm: INSERT: [[{args.ctype}]]')

# check args
if args.ctype == 'self' or args.ctype == 'hist':
    args.n_friends = 0
if args.encoder_type == 'avg':
    assert args.avg_len > 0

# load data
data_path = path(__file__).absolute().parent / 'data' / f'fm2_{args.data}.pkl'
dataset = dataHandler(data_path, batch_size=args.batch_size,\
    friend_type=args.friendtype, n_friends = args.n_friends, device=device)

now = datetime.datetime.now()
log_time = now.strftime("%m%d-%H%M%S")
log_file_name = f'{args.data}-{args.ctype}-{args.c}-{log_time}.txt'
logf = open(path(__file__).absolute().parent / 'logs' / log_file_name,'w')

num_users = dataset.num_users
num_items = dataset.num_items

print(f'Dataset: {args.data}')
print(f'#users {num_users}, #items {num_items}')
for k,v in args.__dict__.items():
    print(f'{str(k)}:  {str(v)}', file = logf, flush=True)
    print(f'{str(k)}:  {str(v)}', flush=True)

model = INSERT(hidden_size = args.dim_hidden, dropout=args.dropout, n_friends=args.n_friends, num_users = num_users,\
     num_items=num_items, encoder_type = args.encoder_type, avg_len=args.avg_len, ctype=args.ctype).to(device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.l2)

# load model
start_iter = 1
model_path = path(__file__).absolute().parent / 'logs' / 'saved_models'
if len(args.load) > 0:
    load_model_files = list(model_path.glob(f'{args.data}-{args.ctype}-{args.load}*.model'))
    if len(load_model_files) != 1:
        print(f'Warning: load files founded not 1: {len(load_model_files)}')
    print(f'loading model from {load_model_files[0]}')
    start_iter = str(load_model_files[0].stem).split('-')[-1]
    model.load_state_dict(torch.load(load_model_files[0],map_location=device))

best_recall5_val = 0
best_recall20_val = 0
best_recall5_tst = 0
best_recall20_tst = 0
best_score_message = ''

patience = 0
for idx_iter in range(int(start_iter),args.num_iter+1):
    print(f'Starting epoch {idx_iter} ...')
    dataset.reset_train_batch()
    model.train()
    num_batch_train = dataset.get_num_remain_batches()

    # train
    idx_batch = 0
    c_batch_size = args.batch_size
    loss_sum = 0
    batch_sum = 0
    pbar = progressbar.ProgressBar(max_value=num_batch_train,prefix='{variables.batch} >> {variables.loss} ||',variables={'batch': '--', 'loss': '--'})
    while not args.eval:
        
        batch_users, cur_sess, cur_sess_len, hist_sess, frds_sess = dataset.get_next_train_batch()
        c_batch_size = len(batch_users)
        if c_batch_size == 0:
            break
        output = model(batch_users, cur_sess, hist_sess, frds_sess, cur_sess_len)
        loss = model.loss(output, cur_sess_len)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        batch_sum += 1
        if idx_batch%10 == 0 and idx_batch > 0:
            pbar.update(idx_batch, batch=f'Batch #{idx_batch}/{num_batch_train}', loss=f'loss: {loss_sum/batch_sum:.5f}')
            batch_sum = 0
            loss_sum = 0
        idx_batch+=1
    pbar.finish()
    # test
    # print_gpu_objects()
    
    if idx_iter%args.eval_every ==0 and idx_iter>5:
        print(f'Starting evaluation...')
        tester_val = Tester(session_length=4, k_list=[5, 10, 20])
        if args.valset:
            tester_tst = Tester(session_length=4, k_list=[5, 10, 20])
        else:
            tester_tst = tester_val
        dataset.reset_test_batch()
        model.eval()
        num_batch_test = dataset.get_num_remain_batches()

        batch_sum = 0
        # eval_stime = time.time()
        idx_batch = 0
        
        pbar = progressbar.ProgressBar(max_value=num_batch_test,prefix='{variables.batch} ||',variables={'batch': '--'})
        while True:
            batch_users, cur_sess, cur_sess_len, hist_sess, frds_sess = dataset.get_next_test_batch()
            c_batch_size = len(batch_users)
            if c_batch_size == 0:
                break
            batch_sum += 1

            pred_seq = model.predict(batch_users, cur_sess, hist_sess, frds_sess, cur_sess_len)[1]
            true_seq = model.get_target(cur_sess, cur_sess_len)
            if batch_sum%2==0:#TODO np.random.rand() > 0.5:
                tester_val.evaluate_batch(pred_seq, true_seq, cur_sess_len)
            else:
                tester_tst.evaluate_batch(pred_seq, true_seq, cur_sess_len)

            if idx_batch%10 == 0 and idx_batch > 0:
                pbar.update(idx_batch, batch=f'Batch #{idx_batch}/{num_batch_test}')
            idx_batch+=1
        pbar.finish()
        score_message_val, recall5_val, recall20_val = tester_val.get_stats()
        score_message_tst, recall5_tst, recall20_tst = tester_tst.get_stats()
        print('validation set:')
        print(score_message_val)
        print('test set:')
        print(score_message_tst)
        print(f'crnt val recall: {recall5_val:.4f}/{recall20_val:.4f}')  
        print(f'best val recall: {best_recall5_val:.4f}/{best_recall20_val:.4f}')  
        print(f'Epoch {idx_iter} ...', file=logf)
        print(score_message_tst, file=logf, flush=True)
        now = datetime.datetime.now()
        print(f'[{now.strftime("%m/%d-%H:%M:%S")}] End Iteration {idx_iter} Patience {patience+1}/{args.patience}', file=logf, flush=True)
        print(f'[{now.strftime("%m/%d-%H:%M:%S")}] End Iteration {idx_iter} Patience {patience+1}/{args.patience}')
        print(f'recall5/20 {recall5_tst:.4f}/{recall20_tst:.4f}', file=logf, flush=True)
        print(f'results saved in {log_file_name}')

        if recall5_val > best_recall5_val or recall20_val > best_recall20_val:
            best_recall5_val = max(recall5_val,best_recall5_val)
            best_recall20_val = max(recall20_val,best_recall20_val)
            best_recall5_tst = recall5_tst
            best_recall20_tst = recall20_tst
            best_score_message = score_message_tst
            patience = 0
            if args.save:
                model_file = model_path / f'{args.data}-{args.ctype}-{log_time}-{idx_iter}.model'
                last_model_files = model_path.glob(f'{args.data}-{args.ctype}-{log_time}-*.model')
                for f in last_model_files:
                    os.remove(f)
                torch.save(model.state_dict(), model_file)
                print(f'model saved in {model_file}')
            
        else:
            patience += 1
        
        if patience == args.patience:
            break
print(f'best recall: {best_recall5_tst:.4f}/{best_recall20_tst:.4f}')     
print(f'best recall: {best_recall5_tst:.4f}/{best_recall20_tst:.4f}', file=logf, flush=True)    
print(best_score_message, file=logf, flush=True)   
if args.notify:
    title = f'''
    SSRS:{args.data}/{best_recall5_tst:.4f}/{best_recall20_tst:.4f}
    '''
    text = f'''
    comment: {args.c}
    data:  {args.data}
    ctype: {args.ctype}
    n_friends: {args.n_friends}
    friendtype:{args.friendtype}

    best test recall: {best_recall5_tst:.4f}/{best_recall20_tst:.4f}
    
    result saved in {log_file_name}'''
    notify(title, text) 
logf.close()


