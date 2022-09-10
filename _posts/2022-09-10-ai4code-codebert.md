---
title: codebert(base) - [1md+20code]’s CLS ranking 0.8412
author: mng
date: 2022-09-10 11:15:03 +0900
categories: [NLP, KAGGLE, AI4CODE]
tags: [nlp, kaggle, codebert]
math: true
sitemap :
  priority : 1.0
---

# Preprocess

기본적인 데이터 추출은 baseline의 그것과 같다

```python
class Preprocessor:
    def __init__(self, **args):
        self.__dict__.update(args)
        self.data_dir = Path(self.input_path)

    def read_notebook(self, path):
        return (
            pd.read_json(path, dtype={'cell_type': 'category', 'source': 'str'})
            .assign(id=path.stem)
            .rename_axis('cell_id')
        )

    def get_ranks(self, base, derived):
        return [base.index(d) for d in derived]

    def run(self, mode='train', nvalid=0.1):
        if os.path.exists(self.train_path) and os.path.exists(self.val_path):
            print('train_df, val_df are already exits')
            train_df = pd.read_csv(self.train_path)
            val_df = pd.read_csv(self.val_path) 
            train_df_mark = pd.read_csv(self.train_mark_path)
            val_df_mark = pd.read_csv(self.val_mark_path)
            return train_df, val_df, train_df_mark, val_df_mark

        paths = list((self.data_dir / mode).glob('*.json'))
        notebooks = [self.read_notebook(path)
                     for path in tqdm(paths, desc=f'{mode} NBs')]

        df = (pd.concat(notebooks)
              .set_index('id', append=True)
              .swaplevel()
              .sort_index(level='id', sort_remaining=False))

        df_orders = pd.read_csv(
            self.data_dir / 'train_orders.csv',
            index_col='id',
            squeeze=True).str.split()

        df_orders_ = df_orders.to_frame().join(
            df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
            how='right'
        )

        ranks = {}
        for id_, cell_order, cell_id in df_orders_.itertuples():
            ranks[id_] = {'cell_id': cell_id,
                          'rank': self.get_ranks(cell_order, cell_id)}

        df_ranks = (
            pd.DataFrame
            .from_dict(ranks, orient='index')
            .rename_axis('id')
            .apply(pd.Series.explode)
            .set_index('cell_id', append=True)
        )

        df_ancestors = pd.read_csv(
            self.data_dir / 'train_ancestors.csv', index_col='id')
        df = df.reset_index().merge(
            df_ranks, on=['id', 'cell_id']).merge(df_ancestors, on=['id'])
        df['pct_rank'] = df['rank'] / \
            df.groupby('id')['cell_id'].transform('count')

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=nvalid, random_state=0)

        train_ind, val_ind = next(splitter.split(df, groups=df['ancestor_id']))
            
        train_df = df.loc[train_ind].reset_index(drop=True)
        val_df = df.loc[val_ind].reset_index(drop=True)

        train_df_mark = train_df[train_df['cell_type'] == 'markdown'].reset_index(drop=True)
        val_df_mark = val_df[val_df['cell_type'] == 'markdown'].reset_index(drop=True)

        train_df_mark.to_csv(self.train_mark_path + f'_fold{i}')
        val_df_mark.to_csv(self.val_mark_path + f'_fold{i}')

        train_df.to_csv(self.train_path + f'_fold{i}')
        val_df.to_csv(self.val_path + f'_fold{i}')

        return train_df, val_df, train_df_mark, val_df_mark
```

여기에 추가적으로 노트북 당 최대 20개의 코드셀과 코드 셀 개수, 마크 다운 셀의 개수를 추출

```python
class _20CodeCellPreprocessor(Preprocessor):
    def __init__(self, **args):
        self.__dict__.update(args)
        super(_20CodeCellPreprocessor, self).__init__(**args)
        
    def clean_code(self, cell):
        return str(cell).replace('\\n', '\n')

    def sample_cells(self, cells, n=20):
        cells = [self.clean_code(cell) for cell in cells]

        if n >= len(cells): # 코드 셀이 20개 이하라면 그냥 반환
            return [cell[:200] for cell in cells]
        else:
            results = []
            step = len(cells) / n # 총 20개의 코드셀이 샘플링 되도록 스텝을 조절
            idx = 0
            while int(np.round(idx) < len(cells)):
                results.append(cells[int(np.round(idx))])
                idx += step
            assert cells[0] in results # 첫번쨰 코드셀은 반드시 들어가야 한다?
            if cells[-1] not in results: # 말전 코드셀은 반드시 들어가야 한다?
                results[-1] = cells[-1]
            return results

    def get_features(self, df):
        features = dict()
        df = df.sort_values('rank').reset_index(drop=True)

        for idx, sub_df in tqdm(df.groupby('id')):
            features[idx] = dict()
            total_md = sub_df[sub_df.cell_type == 'markdown'].shape[0]
            code_sub_df = sub_df[sub_df.cell_type == 'code']
            total_code = code_sub_df.shape[0]
            codes = self.sample_cells(code_sub_df.source.values, 20)
            features[idx]['total_code'] = total_code
            features[idx]['total_md'] = total_md
            features[idx]['codes'] = codes

        # features = {
        #     노트북id: {
        #         'total_code': 코드 셀의 개수,
        #         'total_md': 마크다운 샐의 개수,
        #         'codes': [코드셀0, 코드셀1, ... , 코드셀 19]
        #     },
        #     ...
        # }
        return features

    def run(self):
        train_df, val_df, train_df_mark, val_df_mark = super().run()

        if os.path.exists(self.train_features_path) and os.path.exists(self.val_features_path):
            print('train_fts, val_fts are already exists')
            train_fts = json.load(open(self.train_features_path))
            val_fts = json.load(open(self.val_features_path))
        else:
            train_fts = self.get_features(train_df)
            val_fts = self.get_features(val_df)          
            json.dump(train_fts, open(self.train_features_path,"wt"))
            json.dump(val_fts, open(self.val_features_path,"wt"))    

        return train_df, val_df, train_df_mark, val_df_mark, train_fts, val_fts
```

# Dataset

BERT에 입력할 커스텀 데이터셋을 만들어줘야 한다

하나의 입력 시퀀스 : [마크다운셀 1개, 코드셀 0, 코드셀 1, … , 코드셀 19]

```python
class _20SampleDataset(Dataset):

    def __init__(self, df, model_name_or_path, total_max_len, md_max_len, fts):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len
        self.fts = fts
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        code_inputs = self.tokenizer.batch_encode_plus(
            [str(x) for x in self.fts[row.id]['codes']],
            add_special_tokens=True,
            max_length=23,
            padding='max_length',
            truncation=True
        )

        n_md = self.fts[row.id]['total_md']
        n_code = self.fts[row.id]['total_code']
        if n_md + n_code == 0:
            fts = torch.FloatTensor([0])
        else:
            fts = torch.FloatTensor([n_md / (n_md + n_code)])

        ids = inputs['input_ids']
        for x in code_inputs['input_ids']:
            ids.extend(x[:-1])
        ids = ids[:self.total_max_len]
        if len(ids) != self.total_max_len:
            ids = ids + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(ids))
        ids = torch.LongTensor(ids)

        mask = inputs['attention_mask']
        for x in code_inputs['attention_mask']:
            mask.extend(x[:-1])
        mask = mask[:self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        assert len(ids) == len(mask)

        return ids, mask, fts, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]
```

# Model

위 입력 시퀀스의 [cls]의 표현 벡터를 추출

```python
class _20SampleModel(nn.Module):
    def __init__(self, model_path):
        super(_20SampleModel, self).__init__()
        
        config = AutoConfig.from_pretrained(model_path)
        
        self.model = AutoModel.from_pretrained(model_path)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.top = nn.Linear(config.hidden_size+1, 1) # for train_fts

    def forward(self, ids, mask, fts, labels=None):
        x = self.model(ids, mask)[0]
        x = torch.cat((x[:, 0, :], fts), 1) # [CLS]만 쓸거임.
        
        x1 = self.top(self.dropout1(x))
        x2 = self.top(self.dropout2(x))
        x3 = self.top(self.dropout3(x))
        x4 = self.top(self.dropout4(x))
        x5 = self.top(self.dropout5(x))
        x = (x1 + x2 + x3 + x4 + x5) / 5
        return x
```

# Train

훈련 시 필요한 것들 세팅

```python
def train_setup(args):
    model = _20SampleModel(model_path=args.model_name_or_path)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = args.num_train_steps
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=3e-5, correct_bias=False)
    scheduler = (
        CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=num_train_optimization_steps,
            cycle_mult=1,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_steps=num_train_optimization_steps * 0.2,
            gamma=1.,
            last_epoch=-1
        ))  # Pytorch scheduler

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=0.05*num_train_optimization_steps,
    #     num_training_steps=num_train_optimization_steps,
    # )
    scaler = torch.cuda.amp.GradScaler()

    return model, optimizer, scheduler, scaler

def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()

def validate(model, val_loader):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout, position=0, leave=True)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)

def train(model, train_loader, val_loader, optimizer, scheduler, scaler, val_df, df_orders, args):
    criterion = torch.nn.L1Loss()

    for e in range(args.epoch, 100):
        model.train()

        tbar = tqdm(train_loader, file=sys.stdout, position=0, leave=True)

        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()

            if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(
                f'Epoch {e+1} Loss: {avg_loss} lr: {scheduler.get_lr()}')

        y_val, y_pred = validate(model, val_loader)
        val_df['pred'] = val_df.groupby(['id', 'cell_type'])['rank'].rank(pct=True)
        val_df.loc[val_df['cell_type'] == 'markdown', 'pred'] = y_pred
        y_dummy = val_df.sort_values('pred').groupby('id')['cell_id'].apply(list)
        preds_score = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
        print("Preds score", preds_score)

        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)

        torch.save(model.state_dict(), args.output_path + f'/model_epoch_{e}_{preds_score}.bin')
```

# Result

public score : 0.8412