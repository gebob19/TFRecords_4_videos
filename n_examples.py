#%%
import pathlib
tfrecords_save_path = pathlib.Path.home()/'data/tfrecords/train'
tfrecords = [str(p) for p in list(tfrecords_save_path.iterdir()) if p.suffix == '.tfrecord']

# %%
import tensorflow.compat.v1 as tf 
from tfrecords import parse_example

# turn of tensorflow logging  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# %%
dataset = tf.data.TFRecordDataset(tfrecords)\
    .map(parse_example)\
    .batch(1)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

from tqdm import tqdm 
pbar = tqdm(total=86017)
i = 0
with tf.Session() as sess: 
    try: 
        while True:
            video_data, labels, filenames = sess.run(next_element)
            i += 1 
            pbar.update(1)
    except: 
        pass

print(i)
pbar.close()


