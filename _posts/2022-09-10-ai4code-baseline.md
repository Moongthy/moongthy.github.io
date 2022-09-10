---
title: distillbert(small) - baseline 0.7499
author: mng
date: 2022-09-10 11:15:01 +0900
categories: [NLP, KAGGLE, AI4CODE]
tags: [nlp, kaggle, distillbert]
math: true
sitemap :
  priority : 1.0
---

# Load data

raw data를 dataframe으로 바꿔주는 함수

```python
def get_df(data_dir, num_rows, folder='train'):

    paths_train = list((data_dir / folder).glob('*.json'))[:num_rows]

    notebooks_train = [
        read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
    ]

    df = (
        pd.concat(notebooks_train)
            .set_index('id', append=True)
            .swaplevel()
            .sort_index(level='id', sort_remaining=False)
    )

    return df
```

train dataset의 정답 (마크다운 셀의 순서)를 추출하는 함수

```python
def get_df_orders(data_dir):
    df_orders = pd.read_csv(
        data_dir / 'train_orders.csv',
        index_col='id',
        squeeze=True
    ).str.split()

    return df_orders
```

인수로 받아들인 df의 순서를 구하는 함수

```python
def get_ranks(base, derived):
    return [base.index(d) for d in derived]

def get_df_ranks(df, data_dir):
    df_orders = get_df_orders(data_dir)

    df_orders_ = df_orders.to_frame().join(
        df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
        how='right'
    )

    ranks = {}
    for id_, cell_order, cell_id in df_orders_.itertuples():
        ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

    df_ranks = (
        pd.DataFrame
            .from_dict(ranks, orient='index')
            .rename_axis('id')
            .apply(pd.Series.explode)
            .set_index('cell_id', append=True)
    )

    return df_ranks
```

train_dataset의 부모 노트북(fork from)을 구하는 함수

```python
def get_df_ancestors(data_dir):
    df_ancestors = pd.read_csv(data_dir / 'train_ancestors.csv', index_col='id')
    return df_ancestors
```

df를 train과 valid로 나누는 함수

```python
def get_df_train_valid(df, valid_size, random_state):
    NVALID = valid_size  # size of validation set

    splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)

    train_ind, val_ind = next(splitter.split(df, groups=df["ancestor_id"]))

    train_df = df.loc[train_ind].reset_index(drop=True)
    val_df = df.loc[val_ind].reset_index(drop=True)

    return train_df, val_df
```

대충 전처리하고, train과 valid로 나눈다.

```python
from sklearn.model_selection import GroupShuffleSplit

NUM_TRAIN = 10000

df = get_df(data_dir, num_rows=NUM_TRAIN)# NUM_TRAIN = 10000
df_orders = get_df_orders(data_dir)
df_ranks = get_df_ranks(df, data_dir)
df_ancestors = get_df_ancestors(data_dir)

df = df.reset_index().merge(df_ranks, on=['id', 'cell_id']).merge(df_ancestors, on=['id'])
df['pct_rank'] = df['rank'] / df.groupby('id')['cell_id'].transform('count')

train_df, val_df = get_df_train_valid(df, valid_size=.1, random_state=0)

train_df_mark = train_df[train_df['cell_type'] == 'markdown'].reset_index(drop=True)
val_df_mark = val_df[val_df['cell_type'] == 'markdown'].reset_index(drop=True)
```

# Train

흔한 bert 모델 fine tuning 의 모오습

```python
from torch.utils.data import Dataset

class MarkdownDataset(Dataset):
    def __init__(self, df, max_len, tokenizer):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        ids = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        return ids, mask, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]

class MarkdownModel(nn.Module):
    def __init__(self, distill_bert):
        super(MarkdownModel, self).__init__()
        self.distill_bert = distill_bert
        self.top = nn.Linear(768, 1)

    def forward(self, ids, mask):
        # (32, 128, 768)
        x = self.distill_bert(ids, mask)[0]
        # (32, 1, 768): [CLS]
        x = self.top(x[:, 0, :])
        return x

from torch.utils.data import DataLoader

BS = 32
NW = 8

MAX_LEN = 128

# 1.Get bert & tokenizer
distill_bert = DistilBertModel.from_pretrained(BERT_PATH)
tokenizer = DistilBertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

# 2.Get Dataset
train_ds = MarkdownDataset(train_df_mark, max_len=MAX_LEN, tokenizer=tokenizer)
val_ds = MarkdownDataset(val_df_mark, max_len=MAX_LEN, tokenizer=tokenizer)

# 3.Get DataLoader
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=NW,
                          pin_memory=False, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=NW,
                        pin_memory=False, drop_last=False)
```

간단한 스케줄러, 옵티마이저 등등 필요한 함수와 validate 및 train 함수

```python
from sklearn.metrics import mean_squared_error

def adjust_lr(optimizer, epoch):
    if epoch < 1:
        lr = 5e-5
    elif epoch < 2:
        lr = 1e-3
    elif epoch < 3:
        lr = 1e-4
    else:
        lr = 1e-5

    for p in optimizer.param_groups:
        p['lr'] = lr
    return lr

def get_optimizer(net):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=3e-4,
                                 betas=(.9, .999),
                                 eps=1e-08)
    return optimizer

# divide input and target from train_loader
def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()

def validate(model, val_loader):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            pred = model(inputs[0], inputs[1])

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)

def train(model, train_loader, val_loader, epochs):
    np.random.seed(0)

    optimizer = get_optimizer(model)

    criterion = torch.nn.MSELoss()

    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)

        lr = adjust_lr(optimizer, e)

        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            # inputs = (ids, mask)
            # target = torch.FloatTensor([row.pct_rank])
            inputs, target = read_data(data)

            optimizer.zero_grad()
            pred = model(inputs[0], inputs[1])

            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f'Epoch {e+1} Loss: {avg_loss} lr: {lr}')

        y_val, y_pred = validate(model, val_loader)

        print('Validation MSE: ', np.round(mean_squared_error(y_val, y_pred), 4))
        print()
    return model, y_pred
```

```python
model = MarkdownModel(distill_bert)
model = model.cuda()
model, y_pred = train(model, train_loader, val_loader, epochs=1)
```

Epoch 1 Loss: 0.0584 lr: 5e-05: 100% 4439/4439 [14:54<00:00, 5.25it/s] 

100% 461/461 [00:41<00:00, 16.33it/s]

Validation MSE: 0.051

# Evaluate

```python
from bisect import bisect

def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions

def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0  # twice the maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max

val_df['pred'] = val_df.groupby(['id', 'cell_type'])['rank'].rank(pct=True)
val_df.loc[val_df['cell_type'] == 'markdown', 'prde'] = y_pred

y_dummy = val_df.sort_values('pred').groupby('id')['cell_id'].apply(list)
kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
```

`0.9221784760258563`

```python
test_df = get_df(data_dir, num_rows=None, folder='test').reset_index()

test_df["rank"] = test_df.groupby(["id", "cell_type"]).cumcount()
test_df["pred"] = test_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)

test_df["pct_rank"] = 0
test_ds = MarkdownDataset(test_df[test_df["cell_type"] == "markdown"].reset_index(drop=True),
                          max_len=MAX_LEN,
                          tokenizer=tokenizer)
test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=NW,
                          pin_memory=False, drop_last=False)

len(test_ds), test_ds[0]
```

Train NBs: 100% 4/4 [00:00<00:00, 44.16it/s]

```
(43,
 (tensor([  101,  1001, 25169,  2951,   100,  2292,  1005,  1055,  4094,  1996,
           2951,  2061,  7473,  2050,  2064,  2022,  4162,   102,     0,     0,
              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
              0,     0,     0,     0,     0,     0,     0,     0]),
  tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0]),
  tensor([0.])))
```

```python
_, y_test = validate(model, test_loader)
```

100% 2/2 [00:01<00:00, 1.13it/s]

```python
test_df.loc[test_df["cell_type"] == "markdown", "pred"] = y_test
test_df
```

# Submit

```python
sub_df = test_df.sort_values("pred").groupby("id")["cell_id"].apply(lambda x: " ".join(x)).reset_index()
sub_df.rename(columns={"cell_id": "cell_order"}, inplace=True)
sub_df.head()
```

# Result

Run: 1068.6s - GPU

Public score : 0.7499

# Link

[](https://www.kaggle.com/code/mungeunjo/ai4code-02-distilbert-baseline?scriptVersionId=99014532)