import os
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import torch
import bisect
import warnings

class Dataset_Stock(Dataset):
    def __init__(self, file, split='train', size=None,
                 features='M', 
                 target='OT', scale=True, timeenc=1, freq='b', use_time_features=False, train_split=0.7, test_split=0.1) -> None:
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 30 * 4
            self.label_len = 7 * 4
            self.pred_len = 1 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.file = file
        self.use_time_features = use_time_features

        self.train_split, self.test_split = train_split, test_split
        #self.stock_codes = []
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()


        # Get a list of all parquet files in the folder
        file = self.file


        
        # Read each file into a dataframe and append to the list
        
        df_raw = pd.read_csv(file)
        
        #df_raw.rename(columns={0:'date',1:'code'}, inplace=True)
        if self.features == 'M' or self.features == 'MS':
            #start_id = 2 + 768*2 
            cols_data = ['close']#['open','high','low','close','pre_close','change','pct_chg','vol','amount'] #df_raw.columns[start_id:] # ignore date code_idx column and embeddings
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        divided_cnt = len(df_data)/(self.seq_len+self.pred_len)
        remainder = len(df_data) % (self.seq_len+self.pred_len)
        if divided_cnt < 1:
            self.data_x = []
            self.data_y = []
            self.data_stamp = []

            self.data_x_txts = []
            self.data_y_txts = []
            return
        else:
            all_cnt = len(df_data) - self.seq_len - self.pred_len + 1

        # emb_start_at = 2
        # emb_end_at = 2+768*2
        texts_columns = ['title','key_note']#df_raw.columns[emb_start_at:emb_end_at]
        texts = df_raw[texts_columns]
        #df_embeddings = torch.Tensor(df_raw[emb_columns].values).reshape(-1,2,768)

        num_train = int(all_cnt * self.train_split)
        num_test = int(all_cnt * self.test_split)
        num_valid = all_cnt - num_train - num_test
        #print(f'input sequence length: {len(df_data)}')
        # if num_valid - self.seq_len <= 0 :
        #         self.data_x = []
        #         self.data_y = []
        #         return 
        # if self.set_type == 1:
        #     if num_valid -self.seq_len <= 0 :
        #         self.data_x = []
        #         self.data_y = []
        # valid_start = len(df_raw)  - self.seq_len - 2*self.pred_len
        # if valid_start < 0:
        #     valid_start = 0
            
        # test_start = len(df_data) - num_test - self.seq_len
        # if test_start < 0:
        #     test_start = len(df_data) - self.seq_len
        
        train_start = 0
        train_end = train_start if num_train == 0 else (num_train-1) + (self.seq_len + self.pred_len)

        valid_start = num_train  if train_end+1 < len(df_raw) else train_end
        valid_end = valid_start if num_valid == 0 else  valid_start + (num_valid-1) + (self.seq_len + self.pred_len)

        test_start = num_train + num_valid if valid_end+1 < len(df_raw) else valid_end
        test_end = test_start if num_test == 0 else test_start + (num_test-1) + (self.seq_len + self.pred_len)

        border1s = [0, valid_start ,  test_start]
        border2s = [train_end, valid_end, test_end]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
       
        #df_code_idx = df_raw[['idx']] #[border1:border2]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
            #data = np.hstack((data,df_raw['code'].values.astype(int).reshape(len(df_raw['code']),1)))
        else:
            data = df_data.to_numpy()

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        self.data_x_txts = texts[border1:border2]
        self.data_y_txts = texts[border1:border2]
        #self.code_idx = df_code_idx

        #self.code = df_code_idx.iloc[0].values
        #print(self.code )

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_txts = self.data_x_txts[s_begin:s_end]
        seq_y_txts = self.data_y_txts[r_begin:r_end]
        # seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        # seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        #seq_x_emb['news_emb'].apply(lambda x: json.loads(x))
        seq_x = torch.Tensor(seq_x)
        seq_y = torch.Tensor(seq_y)
        seq_x_mark = torch.Tensor(seq_x_mark)
        seq_y_mark = torch.Tensor(seq_y_mark)
        # seq_x_emb = torch.Tensor(seq_x_emb)
        # seq_y_emb = torch.Tensor(seq_y_emb)
        
        if self.use_time_features: return ((seq_x, seq_y, seq_x_mark, seq_y_mark), (seq_x_txts, seq_y_txts)) #, seq_x_emb, seq_y_emb) # _torch(seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_emb, seq_y_emb)
        else: return (seq_x, seq_y)
    
    def __len__(self):
        length = len(self.data_x) - self.seq_len - self.pred_len + 1
        if length < 0:
            return 0
        return length #len(self.data_x) - self.seq_len - self.pred_len + 1
    
class ConcatStockDataset(ConcatDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='*.csv',
                 target='OT', scale=False, timeenc=1, freq='d',
                 time_col_name='date', use_time_features=True, 
                 percent=100,seasonal_patterns=None,
                 train_split=0.8, test_split=0.1) -> None:
        split = flag
        # load files from root_path use pathlib
        files = list(Path(root_path).glob(data_path))
        datasets = []
        for file in files:
            datasets.append(Dataset_Stock(file=file, split = split, size = size,
                                           features=features, target = target, scale = scale, 
                                          timeenc = timeenc, freq = freq,
                                          train_split=train_split, test_split=test_split, use_time_features=use_time_features))


        ConcatDataset.__init__(self, datasets)
        self.scale = scale
    
    def get_scaler(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].scaler
        #return self.datasets[idx].scaler