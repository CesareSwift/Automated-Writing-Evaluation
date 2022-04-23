import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm
import pandas as pd
from tensorflow import keras
from transformers import *
from tensorflow.data import Dataset
MODEL_NAME = "../input/huggingface-bert-variants/bert-base-cased/bert-base-cased"
TRAIN_CSV = "../input/feedback-prize-2021/train.csv"
TRAIN_DIR = "../input/feedback-prize-2021/train/"

# Data Preprocessing

train = pd.read_csv(TRAIN_CSV)
IDS = train.id.unique()
print(f"Number of train samples are: {len(IDS)}")


MAX_LEN = 512 # BERT limit

# THE TOKENS AND ATTENTION ARRAYS
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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

for id_num in tqdm(range(len(IDS))):

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

train_len = [x for x in train_lens if x <=512]
print(f"{len(train_len)/len(IDS) * 100}% of train data has equal or less than 512 tokens")

targets = np.zeros((len(IDS),MAX_LEN,15), dtype='int32')
for k in range(7):
    targets[:,:,2*k] = targets_b[k]
    targets[:,:,2*k+1] = targets_i[k]
targets[:,:,14] = 1-np.max(targets,axis=-1)

np.random.seed(42)
train_idx = np.random.choice(np.arange(len(IDS)),int(0.95*len(IDS)),replace=False)
valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)
np.random.seed(None)
print('Train size',len(train_idx),', Valid size',len(valid_idx))

model = keras.models.load_model("../input/bert-weights/model.h5")
model.summary()

target_map_rev = {0:'Lead', 1:'Position', 2:'Evidence', 3:'Claim', 4:'Concluding Statement',
             5:'Counterclaim', 6:'Rebuttal', 7:'blank'}

p = model.predict([train_tokens[valid_idx,], train_attention[valid_idx,]],
                  batch_size=16, verbose=1)
print('OOF predictions shape:',p.shape)
oof_preds = np.argmax(p,axis=-1)



# VALIDATION AND CV
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

oof = get_preds( dataset='train', verbose=True, text_ids=IDS[valid_idx])
oof.head()


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

valid = train.loc[train['id'].isin(IDS[valid_idx])]


oof['len'] = oof['predictionstring'].apply(lambda x:len(x.split()))
train['len'] = train['predictionstring'].apply(lambda x:len(x.split()))

train.groupby('discourse_type')['len'].describe(percentiles = [0.02, 0.25, 0.50, 0.75, 0.98])


map_clip = {'Lead':9, 'Position':5, 'Evidence':14, 'Claim':3, 'Concluding Statement':11,
             'Counterclaim':6, 'Rebuttal':4}

def threshold(df):
    df = df.copy()
    for key, value in map_clip.items():
    # if df.loc[df['class']==key,'len'] < value
        index = df.loc[df['class']==key].query(f'len<{value}').index
        df.drop(index, inplace = True)
    return df

oof_2 = threshold(oof)
oof_2.head()


f1s = []
CLASSES = oof['class'].unique()
for c in CLASSES:
    pred_df = oof.loc[oof['class']==c].copy()
    gt_df = valid.loc[valid['discourse_type']==c].copy()
    f1 = score_feedback_comp(pred_df, gt_df)
    print(c,f1)
    f1s.append(f1)
print()
print('Overall_before',np.mean(f1s))


f1s = []
CLASSES = oof['class'].unique()
for c in CLASSES:
    pred_df = oof_2.loc[oof_2['class']==c].copy()
    gt_df = valid.loc[valid['discourse_type']==c].copy()
    f1 = score_feedback_comp(pred_df, gt_df)
    print(c,f1)
    f1s.append(f1)
print()
print('Overall_after',np.mean(f1s))


