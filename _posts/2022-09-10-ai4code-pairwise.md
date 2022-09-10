---
title: distillbert(small) - pairwise 0.8171
author: mng
date: 2022-09-10 11:15:02 +0900
categories: [NLP, KAGGLE, AI4CODE]
tags: [nlp, kaggle, distillbert, pairwise]
math: true
sitemap :
  priority : 1.0
---

# Preprocess

이것저것 걸러내기

```python
class PairwisePreprocessor(Preprocessor):
    def __init__(self, **args):
        self.__dict__.update(args)
        super(PairwisePreprocessor, self).__init__(**args)

        nltk.download('wordnet')
        nltk.download('omw-1.4')

        self.stemmer = WordNetLemmatizer()

    def preprocess_text(self, document):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()
        # return document

        # Lemmatization
        tokens = document.split()
        tokens = [self.stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if len(word) > 3]

        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    def run(self):
        if os.path.exists(self.pairwise_train_path) and os.path.exists(self.pairwise_val_path):
            print('pairwise_train_df and val_df already exists')
            train_df = pd.read_csv(self.pairwise_train_path)
            val_df = pd.read_csv(self.pairwise_val_path)
        else:
            print('generate_train_df and val_df')
            train_df, val_df, _, _ = super().run()
        
            train_df.source = train_df.source.apply(self.preprocess_text)
            val_df.source = val_df.source.apply(self.preprocess_text)

            train_df.to_csv(self.pairwise_train_path)
            val_df.to_csv(self.pairwise_val_path)

        if os.path.exists(self.dict_cellid_source_path):
            dict_cellid_source = joblib.load(self.dict_cellid_source_path)
        else:
            df = pd.concat([train_df, val_df])
            dict_cellid_source = dict(zip(df['cell_id'].values, df['source'].values))
            joblib.dump(dict_cellid_source, self.dict_cellid_source_path)
        
        return train_df, val_df, dict_cellid_source
```

# Pairwise dataset

## Triplet

pairwise 이니 triplet:(문장 A, 문장 B, isNext or notNext)들을 준비해준다. 실험 결과 최적 비율은 True:False = 1:9이고, 5:5는 데이터의 개수가 너무 적어진다. 

```python
def generate_triplets(df, args, mode='train'):
    print(f'generate {mode} triplets')

    triplets = []
    drop_sz = 1000 if args.debug else 10000
    random_drop = np.random.random(size=drop_sz) > .9
    count = 0

    for id, df_tmp in tqdm(df.groupby('id')):
        df_tmp_markdown = df_tmp[df_tmp['cell_type'] == 'markdown']
        df_tmp_code = df_tmp[df_tmp['cell_type'] == 'code']

        df_tmp_code_rank = df_tmp_code['rank'].values
        df_tmp_code_cell_id = df_tmp_code['cell_id'].values

        for cell_id, rank in df_tmp_markdown[['cell_id', 'rank']].values:
            # cell_id의 마크다운 바로 뒤에 나오는 코드셀이면 True
            labels = np.array([(r == (rank+1)) for r in df_tmp_code_rank]).astype(int)

            for cid, label in zip(df_tmp_code_cell_id, labels):
                count += 1
                if label == 1 or random_drop[count % drop_sz] or mode=='test':
                    triplets.append([cell_id, cid, label])
    
    return triplets
```

## Pairwise-dataset

```python
class PairwiseDataset(Dataset):
    def __init__(self, df, args):
        self.df = df
        self.max_len = args.total_max_len
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.dict_cellid_source = joblib.load(args.dict_cellid_source_path)

    def __getitem__(self, index):
        row = self.df[index]

        label = row[-1]

        txt = self.dict_cellid_source[row[0]] + \
            '[SEP]' + self.dict_cellid_source[row[1]]

        inputs = self.tokenizer.encode_plus(
            txt,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = torch.LongTensor(inputs['input_ids'])
        mask = torch.LongTensor(inputs['attention_mask'])

        return ids, mask, torch.FloatTensor([label])

    def __len__(self):
        return len(self.df)
```

할거 하고 데이터 로더 반환

```python
def pairwise_data_setup(train_df, val_df, args):
    train_triplets = generate_triplets(train_df, args, mode='train')
    # test 모드는 drop없이 다 때려박기 때문에 데이터 개수가 많다.
    val_triplets = generate_triplets(val_df, args, mode='test') 

    train_ds = PairwiseDataset(train_triplets, args)
    val_ds = PairwiseDataset(val_triplets, args)

    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=False,
                              drop_last=True)

    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=False,
                            drop_last=False)

    return train_loader, val_loader
```

# Train

학습에 필요한 이것저것 준비

```python
def pairwise_train_setup(args):
    model = PairwiseModel(model_path=args.model_name_or_path)
    num_train_optimization_steps = args.num_train_steps
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=3e-4,
                                 betas=(0.9, 0.999),
                                 eps=1e-8)  # 1e-08)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.05*num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )
    scaler = torch.cuda.amp.GradScaler()

    return model, optimizer, scheduler, scaler
```

train 함수 본문

이진 분류 문제로 치환하면서 BCELoss를 사용하게 되었는데, BCELoss는 amp를 못쓴다. 따라서 BCEwithLogitLoss인가 뭔가 그걸 사용하고, 위 모델 아웃풋에도 좀 손대서 amp로 돌렸다. 바꾼 코드는 어딨는지 까먹었고, 찾기 귀찮다. 아래 코드도 잘 돌아간다 (대신 오래걸림)

```python
def pairwise_train(model, train_loader, val_loader, optimizer, scheduler, scaler, val_df, df_orders, args):
    criterion = torch.nn.BCELoss()

    for e in range(args.epoch, args.epochs):
        model.train()

        tbar = tqdm(train_loader, file=sys.stdout, position=0, leave=True)

        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            # with torch.cuda.amp.autocast():
            #     pred = model(*inputs)
            #     loss = criterion(pred, target)
            # scaler.scale(loss).backward()
            optimizer.zero_grad()

            pred = model(*inputs)
            loss = criterion(pred, target)
            loss.backward()

            # scaler.step(optimizer)
            # scaler.update()

            optimizer.step()
            scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(
                f"Epoch {e+1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")

        y_val, y_pred = pairwise_validate(model, val_loader)
        y_pred = get_preds(y_pred, val_df)
        val_df['pred'] = val_df.groupby(['id', 'cell_type'])['rank'].rank(pct=True)
        val_df.loc[val_df['cell_type'] == 'markdown', 'pred'] = y_pred
        y_dummy = val_df.sort_values('pred').groupby('id')['cell_id'].apply(list)
        print("Preds score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))

        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, args.output_path + f'/chekcpoint_{e}.pt')
        torch.save(model.state_dict(), args.output_path + f'/model_epoch_{e}.bin')
```

# Validate

pairwise 예측으로 생긴 로짓을 가지고 다시 순서를 매겨야한다. 조금 골때리는 부분. 원본 코드 작성자의 방법과 다른 방식도 이것저것 시도해 보았으나 큰 성능향상을 느끼지 못하였다.

```python
def validate(model, val_loader, mode='train'):
    model.eval()
    
    tbar = tqdm(val_loader, file=sys.stdout)
    
    preds = np.zeros(len(val_loader.dataset), dtype='float32')
    labels = []
    count = 0

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            pred = model(inputs[0], inputs[1]).detach().cpu().numpy().ravel()

            preds[count:count+len(pred)] = pred
            count += len(pred)
            
            if mode=='test':
              labels.append(target.detach().cpu().numpy().ravel())
    if mode=='test':
      return preds
    else:
      return np.concatenate(labels), np.concatenate(preds)

preds_copy = y_test
pred_vals = []
count = 0
for id, df_tmp in tqdm(test_df.groupby('id')):
  df_tmp_mark = df_tmp[df_tmp['cell_type']=='markdown']
  df_tmp_code = df_tmp[df_tmp['cell_type']!='markdown']
  df_tmp_code_rank = df_tmp_code['rank'].rank().values
  N_code = len(df_tmp_code_rank)
  N_mark = len(df_tmp_mark)

  preds_tmp = preds_copy[count:count+N_mark * N_code]

  count += N_mark * N_code

  for i in range(N_mark):
    pred = preds_tmp[i*N_code:i*N_code+N_code] 

    softmax = np.exp((pred-np.mean(pred)) *20)/np.sum(np.exp((pred-np.mean(pred)) *20)) 

    rank = np.sum(softmax * df_tmp_code_rank)
    pred_vals.append(rank)

del model
del test_triplets[:]
del dict_cellid_source
gc.collect()
```

# Result

public score - 0.8171