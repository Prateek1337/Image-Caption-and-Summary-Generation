from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard

from pycocotools.coco import COCO
# import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image

from dv_utils import *


dataDir='.'
dataType='train2017'

batch_size = 128
embedding_size = 128
cell_state_size = 512
num_words = 10000
epochs = 3

capsAnnFile='{}/annotations/captions_{}.json'.format(dataDir,dataType)
instAnnFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco_caps=COCO(capsAnnFile)
coco_inst=COCO(instAnnFile)

activation_file = h5py.File('vgg_activations.h5','r')
vgg_activations = activation_file['last_layer_activations'][:]

id_list = getAllIds(coco_inst)

start = 'ssstrt '
end = ' eenddd'

captions = get_captions(id_list,coco_caps)
captions_marked = [[str(start)+str(info)+str(end) for info in cap] for cap in captions]
captions_train = [sent for cap in captions_marked for sent in cap]

tokenizer = TokenizerExt(captions_train)
train_tokens = tokenizer.captions_to_tokens(captions_marked)

num_images = vgg_activations.shape[0]
id_map = np.array(range(num_images))

steps = int(len(captions_train)/batch_size)
activation_vector_length = vgg_activations.shape[1]
del captions
del captions_train
del captions_marked
del coco_inst
del coco_caps

image_activation_input = Input(shape=(activation_vector_length,),name='img_act_input')

model_map_layer = Dense(cell_state_size,activation='tanh',name='fc_map')(image_activation_input)

lang_model_input = Input(shape=(None,),name="lang_input")
lang_embed = Embedding(input_dim=num_words,output_dim=embedding_size,name='lang_embed')(lang_model_input)

lang_gru1 = GRU(cell_state_size, name='lang_gru1',return_sequences=True)(lang_embed,initial_state=model_map_layer)
lang_gru2 = GRU(cell_state_size, name='lang_gru2',return_sequences=True)(lang_gru1,initial_state=model_map_layer)
lang_gru3 = GRU(cell_state_size, name='lang_gru3',return_sequences=True)(lang_gru2,initial_state=model_map_layer)

lang_out = Dense(num_words,activation='linear',name='lang_out')(lang_gru3)
language_model = Model(inputs=[image_activation_input,lang_model_input],outputs=[lang_out])


optimizer = RMSprop()
language_model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy')

path_checkpoint = 'train_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,verbose=1,save_weights_only=True)
callback_tensorboard = TensorBoard(log_dir='./train_logs/',histogram_freq=0,write_graph=False)
callbacks = [callback_checkpoint, callback_tensorboard]

generator = data_generator(id_map, vgg_activations, train_tokens, batch_size)
language_model.fit_generator(generator,
                            steps_per_epoch=steps,
                            epochs=epochs,
                            callbacks=callbacks)