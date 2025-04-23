import seqtag_keras

from seqtag_keras.models import BiLSTMCRF
from seqtag_keras.utils import load_glove
from seqtag_keras.trainer import Trainer

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

import os

save_folder = 'test_save'

checkpoint_path = os.path.join(save_folder, 'checkpoint')
final_weights_path = os.path.join(save_folder, 'final_weights')
params_path = os.path.join(save_folder, 'params')
utils_path = os.path.join(save_folder, 'utils')    

checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, mode='max', monitor='f1')
earlystop = EarlyStopping(patience=3, monitor='f1', mode='max')

model = seqtag_keras.Sequence()

x = [
    [
        "a",
        "b",
        "c"
    ],
    [
        "a",
        "b",
        "c"
    ]
] * 100

y = [
    [
        "O",
        "O",
        "B-x"
    ],
    [
        "O",
        "O",
        "B-x"
    ]
] * 100


model.fit(x, y, x_valid=x[:10], y_valid=y[:10], epochs=10, batch_size=4, callbacks=[checkpoint, earlystop])

model.save(final_weights_path, params_path, utils_path)

