import pathlib
from tfrecords import TFRecords4Video

# will save as -- tfrecords_save_path/(train/shard1.tfrecord)
tfrecords_save_path = \
    pathlib.Path.home()/'Documents/gradschool/thesis/data/tfrecords'
# points to dir which includes train.txt, val.txt and test.txt
datafile_path = tfrecords_save_path 
# datafile_prefix/filepath for all 'filepath label' in train.txt
datafile_prefix = pathlib.Path.home()/'Documents/gradschool/thesis/data'\
    '/something-something/20bn-something-something-v1'

tfrv = TFRecords4Video(tfrecords_save_path, datafile_path, datafile_prefix, 'images')
tfrv.create_tfrecords()