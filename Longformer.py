import os
# DECLARE HOW MANY GPUS YOU WISH TO USE. 
# KAGGLE ONLY HAS 1, BUT OFFLINE, YOU CAN USE MORE
os.environ["CUDA_VISIBLE_DEVICES"]="0" #0,1,2,3 for four gpu

# VERSION FOR SAVING/LOADING MODEL WEIGHTS
# THIS SHOULD MATCH THE MODEL IN LOAD_MODEL_FROM
VER=14

# IF VARIABLE IS NONE, THEN NOTEBOOK COMPUTES TOKENS
# OTHERWISE NOTEBOOK LOADS TOKENS FROM PATH
LOAD_TOKENS_FROM = '../input/tf-longformer-v12'

# IF VARIABLE IS NONE, THEN NOTEBOOK TRAINS A NEW MODEL
# OTHERWISE IT LOADS YOUR PREVIOUSLY TRAINED MODEL
LOAD_MODEL_FROM = '../input/tflongformerv14'

# IF FOLLOWING IS NONE, THEN NOTEBOOK
# USES INTERNET AND DOWNLOADS HUGGINGFACE
# CONFIG, TOKENIZER, AND MODEL
DOWNLOADED_MODEL_PATH = '../input/tf-longformer-v12'

if DOWNLOADED_MODEL_PATH is None:
    DOWNLOADED_MODEL_PATH = 'model'
MODEL_NAME = 'allenai/longformer-base-4096'

if DOWNLOADED_MODEL_PATH == 'model':
    os.mkdir('model')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained('model')

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.save_pretrained('model')

    backbone = TFAutoModel.from_pretrained(MODEL_NAME, config=config)
    backbone.save_pretrained('model')


# Load Libraries

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import *
print('TF version',tf.__version__)

# USE MULTIPLE GPUS
if os.environ["CUDA_VISIBLE_DEVICES"].count(',') == 0:
    strategy = tf.distribute.get_strategy()
    print('single strategy')
else:
    strategy = tf.distribute.MirroredStrategy()
    print('multiple strategy')

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
print('Mixed precision enabled')

train = pd.read_csv('../input/feedback-prize-2021/train.csv')
print( train.shape )
train.head()

print('The train labels are:')
train.discourse_type.unique()

IDS = train.id.unique()
print('There are',len(IDS),'train texts.')

# Tokenize Train


MAX_LEN = 1024

# THE TOKENS AND ATTENTION ARRAYS
tokenizer = AutoTokenizer.from_pretrained(DOWNLOADED_MODEL_PATH)
train_tokens = np.zeros((len(IDS),MAX_LEN), dtype='int32')
train_attention = np.zeros((len(IDS),MAX_LEN), dtype='int32')

# THE 14 CLASSES FOR NER
lead_b = np.zeros((len(IDS),MAX_LEN))
lead_i = np.zeros((len(IDS),MAX_LEN))

position_b = np.zeros((len(IDS),MAX_LEN))
position_i = np.zeros((len(IDS),MAX_LEN))

evidence_b = np.zeros((len(IDS),MAX_LEN))
evidence_i = np.zeros((len(IDS),MAX_LEN))

claim_b = np.zeros((len(IDS),MAX_LEN))
claim_i = np.zeros((len(IDS),MAX_LEN))

conclusion_b = np.zeros((len(IDS),MAX_LEN))
conclusion_i = np.zeros((len(IDS),MAX_LEN))

counterclaim_b = np.zeros((len(IDS),MAX_LEN))
counterclaim_i = np.zeros((len(IDS),MAX_LEN))

rebuttal_b = np.zeros((len(IDS),MAX_LEN))
rebuttal_i = np.zeros((len(IDS),MAX_LEN))

# HELPER VARIABLES
train_lens = []
targets_b = [lead_b, position_b, evidence_b, claim_b, conclusion_b, counterclaim_b, rebuttal_b]
targets_i = [lead_i, position_i, evidence_i, claim_i, conclusion_i, counterclaim_i, rebuttal_i]
target_map = {'Lead':0, 'Position':1, 'Evidence':2, 'Claim':3, 'Concluding Statement':4,
             'Counterclaim':5, 'Rebuttal':6}

# WE ASSUME DATAFRAME IS ASCENDING WHICH IT IS
assert (np.sum(train.groupby('id')['discourse_start'].diff() <= 0) == 0)

# FOR LOOP THROUGH EACH TRAIN TEXT
for id_num in range(len(IDS)):
    if LOAD_TOKENS_FROM: break
    if id_num % 100 == 0: print(id_num, ', ', end='')

    # READ TRAIN TEXT, TOKENIZE, AND SAVE IN TOKEN ARRAYS
    n = IDS[id_num]
    name = f'../input/feedback-prize-2021/train/{n}.txt'
    txt = open(name, 'r').read()
    train_lens.append(len(txt.split()))
    tokens = tokenizer.encode_plus(txt, max_length=MAX_LEN, padding='max_length',
                                   truncation=True, return_offsets_mapping=True)
    train_tokens[id_num,] = tokens['input_ids']
    train_attention[id_num,] = tokens['attention_mask']

    # FIND TARGETS IN TEXT AND SAVE IN TARGET ARRAYS
    offsets = tokens['offset_mapping']
    offset_index = 0
    df = train.loc[train.id == n]
    for index, row in df.iterrows():
        a = row.discourse_start
        b = row.discourse_end
        if offset_index > len(offsets) - 1:
            break
        c = offsets[offset_index][0]
        d = offsets[offset_index][1]
        beginning = True
        while b > c:
            if (c >= a) & (b >= d):
                k = target_map[row.discourse_type]
                if beginning:
                    targets_b[k][id_num][offset_index] = 1
                    beginning = False
                else:
                    targets_i[k][id_num][offset_index] = 1
            offset_index += 1
            if offset_index > len(offsets) - 1:
                break
            c = offsets[offset_index][0]
            d = offsets[offset_index][1]


if LOAD_TOKENS_FROM is None:
    plt.hist(train_lens,bins=100)
    plt.title('Histogram of Train Word Counts',size=16)
    plt.xlabel('Train Word Count',size=14)
    plt.show()


if LOAD_TOKENS_FROM is None:
    targets = np.zeros((len(IDS),MAX_LEN,15), dtype='int32')
    for k in range(7):
        targets[:,:,2*k] = targets_b[k]
        targets[:,:,2*k+1] = targets_i[k]
    targets[:,:,14] = 1-np.max(targets,axis=-1)

if LOAD_TOKENS_FROM is None:
    np.save(f'targets_{MAX_LEN}', targets)
    np.save(f'tokens_{MAX_LEN}', train_tokens)
    np.save(f'attention_{MAX_LEN}', train_attention)
    print('Saved NER tokens')
else:
    targets = np.load(f'{LOAD_TOKENS_FROM}/targets_{MAX_LEN}.npy')
    train_tokens = np.load(f'{LOAD_TOKENS_FROM}/tokens_{MAX_LEN}.npy')
    train_attention = np.load(f'{LOAD_TOKENS_FROM}/attention_{MAX_LEN}.npy')
    print('Loaded NER tokens')


# Build Model

def build_model():
    tokens = tf.keras.layers.Input(shape=(MAX_LEN,), name='tokens', dtype=tf.int32)
    attention = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention', dtype=tf.int32)

    config = AutoConfig.from_pretrained(DOWNLOADED_MODEL_PATH + '/config.json')
    backbone = TFAutoModel.from_pretrained(DOWNLOADED_MODEL_PATH + '/tf_model.h5', config=config)

    x = backbone(tokens, attention_mask=attention)
    x = tf.keras.layers.Dense(256, activation='relu')(x[0])
    x = tf.keras.layers.Dense(15, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=[tokens, attention], outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                  loss=[tf.keras.losses.CategoricalCrossentropy()],
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model
with strategy.scope():
    model = build_model()

# Train or Load Model

# LEARNING RATE SCHEDULE AND MODEL CHECKPOINT
EPOCHS = 5
BATCH_SIZE = 4
LRS = [0.25e-4, 0.25e-4, 0.25e-4, 0.25e-4, 0.25e-5]
def lrfn(epoch):
    return LRS[epoch]
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
# TRAIN VALID SPLIT 90% 10%
np.random.seed(42)
train_idx = np.random.choice(np.arange(len(IDS)),int(0.9*len(IDS)),replace=False)
valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)
np.random.seed(None)
print('Train size',len(train_idx),', Valid size',len(valid_idx))

# LOAD MODEL
if LOAD_MODEL_FROM:
    model.load_weights(f'{LOAD_MODEL_FROM}/long_v{VER}.h5')

# OR TRAIN MODEL
else:
    model.fit(x=[train_tokens[train_idx,], train_attention[train_idx,]],
              y=targets[train_idx,],
              validation_data=([train_tokens[valid_idx,], train_attention[valid_idx,]],
                               targets[valid_idx,]),
              callbacks=[lr_callback],
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=2)

    # SAVE MODEL WEIGHTS
    model.save_weights(f'long_v{VER}.h5')

p = model.predict([train_tokens[valid_idx,], train_attention[valid_idx,]],
                  batch_size=16, verbose=2)
print('OOF predictions shape:',p.shape)
oof_preds = np.argmax(p,axis=-1)

target_map_rev = {0:'Lead', 1:'Position', 2:'Evidence', 3:'Claim', 4:'Concluding Statement',
             5:'Counterclaim', 6:'Rebuttal', 7:'blank'}


def get_preds(dataset='train', verbose=True, text_ids=IDS[valid_idx], preds=oof_preds):
    all_predictions = []

    for id_num in range(len(preds)):

        # GET ID
        if (id_num % 100 == 0) & (verbose):
            print(id_num, ', ', end='')
        n = text_ids[id_num]

        # GET TOKEN POSITIONS IN CHARS
        name = f'../input/feedback-prize-2021/{dataset}/{n}.txt'
        txt = open(name, 'r').read()
        tokens = tokenizer.encode_plus(txt, max_length=MAX_LEN, padding='max_length',
                                       truncation=True, return_offsets_mapping=True)
        off = tokens['offset_mapping']

        # GET WORD POSITIONS IN CHARS
        w = []
        blank = True
        for i in range(len(txt)):
            if (txt[i] != ' ') & (txt[i] != '\n') & (txt[i] != '\xa0') & (txt[i] != '\x85') & (blank == True):
                w.append(i)
                blank = False
            elif (txt[i] == ' ') | (txt[i] == '\n') | (txt[i] == '\xa0') | (txt[i] == '\x85'):
                blank = True
        w.append(1e6)

        # MAPPING FROM TOKENS TO WORDS
        word_map = -1 * np.ones(MAX_LEN, dtype='int32')
        w_i = 0
        for i in range(len(off)):
            if off[i][1] == 0: continue
            while off[i][0] >= w[w_i + 1]: w_i += 1
            word_map[i] = int(w_i)

        # CONVERT TOKEN PREDICTIONS INTO WORD LABELS
        ### KEY: ###
        # 0: LEAD_B, 1: LEAD_I
        # 2: POSITION_B, 3: POSITION_I
        # 4: EVIDENCE_B, 5: EVIDENCE_I
        # 6: CLAIM_B, 7: CLAIM_I
        # 8: CONCLUSION_B, 9: CONCLUSION_I
        # 10: COUNTERCLAIM_B, 11: COUNTERCLAIM_I
        # 12: REBUTTAL_B, 13: REBUTTAL_I
        # 14: NOTHING i.e. O
        ### NOTE THESE VALUES ARE DIVIDED BY 2 IN NEXT CODE LINE
        pred = preds[id_num,] / 2.0

        i = 0
        while i < MAX_LEN:
            prediction = []
            start = pred[i]
            if start in [0, 1, 2, 3, 4, 5, 6, 7]:
                prediction.append(word_map[i])
                i += 1
                if i >= MAX_LEN: break
                while pred[i] == start + 0.5:
                    if not word_map[i] in prediction:
                        prediction.append(word_map[i])
                    i += 1
                    if i >= MAX_LEN: break
            else:
                i += 1
            prediction = [x for x in prediction if x != -1]
            if len(prediction) > 4:
                all_predictions.append((n, target_map_rev[int(start)],
                                        ' '.join([str(x) for x in prediction])))

    # MAKE DATAFRAME
    df = pd.DataFrame(all_predictions)
    df.columns = ['id', 'class', 'predictionstring']

    return df


oof = get_preds( dataset='train', verbose=True, text_ids=IDS[valid_idx] )
oof.head()


print('The following classes are present in oof preds:')
oof['class'].unique()



# Compute Validation Metric

# CODE FROM : Rob Mulla @robikscube
# https://www.kaggle.com/robikscube/student-writing-competition-twitch
def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id', 'discourse_type', 'predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df = pred_df[['id', 'class', 'predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on=['id', 'class'],
                           right_on=['id', 'discourse_type'],
                           how='outer',
                           suffixes=('_pred', '_gt')
                           )
    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])

    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1', 'overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP') \
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id', 'predictionstring_gt']).first()['pred_id'].values

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    return my_f1_score


# VALID DATAFRAME
valid = train.loc[train['id'].isin(IDS[valid_idx])]


f1s = []
CLASSES = oof['class'].unique()
for c in CLASSES:
    pred_df = oof.loc[oof['class']==c].copy()
    gt_df = valid.loc[valid['discourse_type']==c].copy()
    f1 = score_feedback_comp(pred_df, gt_df)
    print(c,f1)
    f1s.append(f1)
print()
print('Overall',np.mean(f1s))


