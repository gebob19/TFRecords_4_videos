import pathlib
from tfrecords import TFRecords4Video

# will save as -- tfrecords_save_path/(train/shard1.tfrecord)
tfrecords_save_path = pathlib.Path.home()/'data/tfrecords/'
# points to dir which includes train.txt, val.txt and test.txt
datafile_path = tfrecords_save_path 
# datafile_prefix/filepath for all 'filepath label' in train.txt
datafile_prefix = pathlib.Path.home()/'data/something-something/20bn-something-something-v1'

# turn off tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

tfrv = TFRecords4Video(tfrecords_save_path, datafile_path, datafile_prefix, 'images') 
tfrv.split2records('train', max_bytes=1e8)
tfrv.split2records('val', max_bytes=1e8)

# paths, labels = tfrv.extract_pathlabels('train')
# paths, labels = paths[:100], labels[:100]
# tfrv.pathlabels2records(paths, labels, 'train', 1e8)