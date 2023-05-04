from model import model
from utils import preprocess, hms, postprocessor
import tensorflow as tf
import os
import time
import csv

# define the train and val data
train_data = preprocess('D:/En-for-MOTION/data/train.txt')
train = train_data.copy()

# name the model
model_name = 'model3_new_data_balanced_emotion'

# define the model path
saved_model_path = f"results/{model_name}"

# setting callbacks
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=3, 
                                                verbose=1,
                                                restore_best_weights=True)
csv_log_callback = tf.keras.callbacks.CSVLogger(f'{saved_model_path}/training.csv')

# hyper-parameters
optimizer = tf.optimizers.Adam(0.001)
loss=tf.losses.BinaryCrossentropy(from_logits=True)
metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')]
epochs=10
batch_size=256
callbacks=[es_callback,csv_log_callback]

# model building
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)

# print model architecture
print(model.summary())

# make results folder for this model if does not exist yet
if not os.path.isdir(f"results/{model_name}"):
    os.mkdir(f"results/{model_name}")

# save model summary to txt file
with open(f'{saved_model_path}/modelsummary.txt', 'w') as f:

    model.summary(print_fn=lambda x: f.write(x + '\n'))

# measure training time
t0 = time.time()

# train the model
history = model.fit(train.text,
                    train.hos,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose = 1,
                    shuffle=True)

# make results folder if does not exist yet
if not os.path.isdir("results"):
    os.mkdir("results")

# save the entire model
model.save(saved_model_path)
training_time=hms(time.time()-t0)
print(f'-----------------------------------------\nTraining complete.\nTraining time: {training_time}\nFile saved to {saved_model_path}')

# load the test data
test = preprocess('D:/En-for-MOTION/data/test.txt')

# evaluate the model
loss, acc = model.evaluate(test['text'], test['hos'])
acc = '{:5.2f}%'.format(100 * acc)
loss = '{:5.2f}%'.format(100 * loss)
print(f'-----------------------------------------\nEvaluating the model')
print(f'Restored model, accuracy: {acc}')
print(f'Restored model, loss: {loss}')
print(f'Restored model, training time: {training_time}')

# saving evaluation into csv file in the savde model path
with open(f'{saved_model_path}/evaluation.csv','w',newline='') as file:
    writer = csv.writer(file)

    writer.writerow(["Acc","Loss","Train Time","Epochs","Batch Size"])
    writer.writerow([acc,loss,training_time,epochs, batch_size])