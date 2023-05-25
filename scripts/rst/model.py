import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
from tqdm.notebook import tqdm
import numpy as np
import math
import time
from utils import hms

def hms(seconds):
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = int(seconds % 3600 % 60)
    return f'{h}:{m}:{s}'

df_raw = pd.read_csv('D:/En-for-MOTION/data/final_datasets/discourse_balanced.csv')
label_names = df_raw.discourse_relation.sort_values().unique()
print(label_names)

df_raw['label']=pd.factorize(df_raw['discourse_relation'],sort=True)[0]+1
df_factorized = df_raw[['text','label']]
train_text, test_text, train_label, test_label = train_test_split(df_factorized['text'],df_factorized['label'], test_size=0.2, random_state=42)
train_text, valid_text, train_label, valid_label = train_test_split(train_text, train_label, test_size=0.25, random_state=42)

train_label.value_counts()
valid_label.value_counts()
test_label.value_counts()

# tokenization
PRETRAINED_LM = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM, do_lower_case=True)
tokenizer

# function for encoding
def encode(docs):
    '''
    This function takes list of texts and returns input_ids and attention_mask of texts
    '''
    encoded_dict = tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=128, padding='max_length',
                            return_attention_mask=True, truncation=True, return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

# get input ids and attention masks of the datasets
train_input_ids, train_att_masks = encode(train_text.values.tolist())
valid_input_ids, valid_att_masks = encode(valid_text.values.tolist())
test_input_ids, test_att_masks = encode(test_text.values.tolist())

# creating Datasets and DataLoaders
# the labels into tensors

train_y = torch.LongTensor(train_label.values.tolist())
valid_y = torch.LongTensor(valid_label.values.tolist())
test_y = torch.LongTensor(test_label.values.tolist())
train_y.size(),valid_y.size(),test_y.size()

# create dataloaders for training
BATCH_SIZE = 16
train_dataset = TensorDataset(train_input_ids, train_att_masks, train_y)
train_y
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_dataset = TensorDataset(valid_input_ids, valid_att_masks, valid_y)
valid_sampler = SequentialSampler(valid_dataset)
valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE)

test_dataset = TensorDataset(test_input_ids, test_att_masks, test_y)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

# building model
N_labels = len(train_label.unique())
model = BertForSequenceClassification.from_pretrained(PRETRAINED_LM,
                                                      num_labels=N_labels,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

device = torch.device("cpu")
device
model = model.to(device)

# fine-tuning
# optimizer and scheduler
EPOCHS = 10
LEARNING_RATE = 2e-6

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, 
             num_warmup_steps=0,
            num_training_steps=len(train_dataloader)*EPOCHS )

# training loop
train_loss_per_epoch = []
val_loss_per_epoch = []

# measure training time
t0 = time.time()

for epoch_num in range(EPOCHS):
    print('Epoch: ', epoch_num + 1)
    '''
    Training
    '''
    model.train()
    train_loss = 0
    for step_num, batch_data in enumerate(tqdm(train_dataloader,desc='Training')):
        input_ids, att_mask, labels = [data.to(device) for data in batch_data]
        print(labels)
        output = model(input_ids = input_ids, attention_mask=att_mask, labels= labels)
        print(output)
        
        loss = output.loss
        train_loss += loss.item()

        model.zero_grad()
        loss.backward()
        del loss

        clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    train_loss_per_epoch.append(train_loss / (step_num + 1))              

    '''
    Validation
    '''
    model.eval()
    valid_loss = 0
    valid_pred = []
    with torch.no_grad():
        for step_num_e, batch_data in enumerate(tqdm(valid_dataloader,desc='Validation')):
            input_ids, att_mask, labels = [data.to(device) for data in batch_data]
            output = model(input_ids = input_ids, attention_mask=att_mask, labels= labels)

            loss = output.loss
            valid_loss += loss.item()
   
            valid_pred.append(np.argmax(output.logits.cpu().detach().numpy(),axis=-1))
        
    val_loss_per_epoch.append(valid_loss / (step_num_e + 1))
    valid_pred = np.concatenate(valid_pred)

    '''
    Loss message
    '''
    print("{0}/{1} train loss: {2} ".format(step_num+1, math.ceil(len(train_text) / BATCH_SIZE), train_loss / (step_num + 1)))
    print("{0}/{1} val loss: {2} ".format(step_num_e+1, math.ceil(len(valid_text) / BATCH_SIZE), valid_loss / (step_num_e + 1)))
    
training_time=hms(time.time()-t0)
print(f"Training completed.\nTraining time: {training_time}")

# performance on validation
print('classifiation report')
print(classification_report(valid_pred, valid_label.to_numpy(), target_names=label_names))

# evaluation on test
model.eval()
test_pred = []
test_loss= 0
with torch.no_grad():
    for step_num, batch_data in tqdm(enumerate(test_dataloader)):
        input_ids, att_mask, labels = [data.to(device) for data in batch_data]
        output = model(input_ids = input_ids, attention_mask=att_mask, labels= labels)

        loss = output.loss
        test_loss += loss.item()
   
        test_pred.append(np.argmax(output.logits.cpu().detach().numpy(),axis=-1))
test_pred = np.concatenate(test_pred)

# performance on test
print('classifiation report')
print(classification_report(test_pred, test_label.to_numpy(),target_names=label_names))

# conf matrix on tst
plot_confusion_matrix(test_pred,test_label.to_numpy(),labels=label_names)

