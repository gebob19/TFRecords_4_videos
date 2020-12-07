
import tensorflow.compat.v1 as tf 
import numpy as np
import imageio 
from tqdm import tqdm
import pathlib

class TFRecords4Video():
    def __init__(self, tfrecords_save_path, datafile_path, datafile_prefix,\
        fn2video):
        """Writes video TFRecords for a given dataset.

        Args:
            tfrecords_save_path (str): Path to save TFRecords to.
            datafile_path (str): Path to find train.txt, val.txt and test.txt
                {train, test, val}.txt file lines are formatted as: 
                    file label

            datafile_prefix (str): Prefix path for files in train.txt. Paths  
            will be given to fn2video function as 'datafile_prefix/file'

            fn2video (str or function): function which takes path and 
            returns the video matrix with size (T, H, W, C). 
            
            Already implemented use cases can use a string: 
            - 'video' for paths which point to a video file (calls vid2numpy)
            - 'images' for paths which point to a folder of images 
            (calls images2numpy)
        """
        self.tfrecords_save_path = pathlib.Path(tfrecords_save_path)
        self.datafile_path = pathlib.Path(datafile_path)
        self.datafile_prefix = pathlib.Path(datafile_prefix)

        # create folders for TFRecord shards 
        tfrecords_save_path.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            (self.tfrecords_save_path/split).mkdir(exist_ok=True)

        # function for fn -> video data (T, H, W, C)
        if fn2video == 'video':
            self.fn2video = vid2numpy
        elif fn2video == 'images':
            self.fn2video = images2numpy
        else: 
            # allow custom parsing of video matrix 
            self.fn2video = fn2video
        
    def format(self, split):
        """Extracts absolute paths and labels from datafiles 
        ({train, val, test}.txt) using 
        self.datafile_path and self.datafile_prefix

        Args:
            split (str): split to get paths from 
            must be a value in {'train', 'test', 'val'}

        Returns:
            tuple(list[pathlib.Path], list[int]): paths and labels from split's
            datafile
        """
        assert split in ['train', 'val', 'test'], "Invalid Split"

        splitfile_path = self.datafile_path/'{}.txt'.format(split)
        assert splitfile_path.exists(), "{} should exist.".format(splitfile_path)

        with open(splitfile_path, 'r') as f:
            lines = f.readlines()

        skip_counter = 0 
        example_paths, example_labels = [], []
        for line in tqdm(lines): 
            fn, label = line.split(' ')
            fn, label = self.datafile_prefix/fn, int(label)
            if pathlib.Path(fn).exists(): 
                example_paths.append(fn)
                example_labels.append(label)
            else: 
                skip_counter += 1 
        print('\nNumber of files not found: {} / {}'.format(skip_counter, len(lines)))
        
        if skip_counter > 0: 
            print('Warning: Some frames were not found, here is an example path \
                to debug: {}'.format(fn))

        return example_paths, example_labels

    def get_example(self, filename, label):
        """Returns a TFRecords example for the given video located at filename 
        with the label label.

        Args:
            filename (pathlib.Path): path to create example from
            label (int): class label for video 

        Returns:
            tf.train.SequenceExample: encoded tfrecord example 
        """
        # read matrix data and save its shape 
        data = self.fn2video(filename)
        t, h, w, c = data.shape

        # save video as list of encoded frames using tensorflow's operation 
        img_bytes = [tf.image.encode_jpeg(frame, format='rgb') for frame in data]
        with tf.Session() as sess: 
            img_bytes = sess.run(img_bytes)
            
        sequence_dict = {}
        # create a feature for each encoded frame
        img_feats = [tf.train.Feature(bytes_list=\
            tf.train.BytesList(value=[imgb])) for imgb in img_bytes]
        # save video frames as a FeatureList
        sequence_dict['video_frames'] = tf.train.FeatureList(feature=img_feats)

        # also store associated meta-data
        context_dict = {}
        context_dict['filename'] = _bytes_feature(str(filename).encode('utf-8'))
        context_dict['label'] = _int64_feature(label)
        context_dict['temporal'] = _int64_feature(t)
        context_dict['height'] = _int64_feature(h)
        context_dict['width'] = _int64_feature(w)
        context_dict['depth'] = _int64_feature(c)

        # combine list + context to create TFRecords example 
        sequence_context = tf.train.Features(feature=context_dict)
        sequence_list = tf.train.FeatureLists(feature_list=sequence_dict)
        example = tf.train.SequenceExample(context=sequence_context, \
            feature_lists=sequence_list)

        return example

    def write(self, paths, labels, split, max_bytes=1e9):
        """Creates TFRecord files in shards from the given path and labels

        Args:
            paths (list[pathlib.Path]): paths of videos to write to TFRecords
            labels (list[int]): labels associated videos
            split (str): datasplit to write to, one of: ('train', 'test', 'val')
            max_bytes (int, optional): approx max size of each shard in bytes.
            Defaults to 1e9.
        """
        assert split in ['train', 'val', 'test'], "Invalid Split"

        shard_count, i = 0, 0
        n_examples = len(paths)

        print('Splitting {} examples into {:.2f} GB shards'.format(\
            n_examples, max_bytes / 1e9))

        pbar = tqdm(total=n_examples)
        while i != n_examples:
            # tf record file to write to 
            tf_record_name = ('{}/shard{}.tfrecord').format(split, shard_count)
            print('\nProcessing: {}'.format(tf_record_name))

            record_file = self.tfrecords_save_path/tf_record_name
            with tf.python_io.TFRecordWriter(str(record_file)) as writer: 
                # split into 1GB (1e9 byte) shards 
                while record_file.stat().st_size < max_bytes and i != n_examples: 
                    # write each example to tfrecord 
                    example_i = tfrv.get_example(train_paths[i], train_labels[i])
                    writer.write(example_i.SerializeToString())
                    # process next example 
                    i += 1 
                    pbar.update(1)

            # process a new shard 
            shard_count += 1
        pbar.close()
        print('Total Number of Shards Created: ', shard_count)

    def split2records(self, split, max_bytes=1e9):
        """Creates TFRecords for a given data split

        Args:
            split (str): split to create for, in ['train', 'test', 'val']
            max_bytes (int, optional): approx max size of each shard in bytes.
            Defaults to 1e9.
        """
        print('Starting processing split {}.'.format(split))

        print('Extracting paths and labels...')
        paths, labels = self.format(split)
        
        print('Writing to TFRecords...')
        self.write(paths, labels, split, max_bytes)
        
        print('Finished processing split {}.'.format(split))

    def create_tfrecords(self, max_bytes=1e9):
        """Creates TFRecords for all splits ('train', 'test', 'val')

        Args:
            max_bytes (int, optional): approx max size of each shard in bytes.
            Defaults to 1e9.
        """
        for split in ['train', 'test', 'val']:
            self.split2records(split, max_bytes)

# TFRecords helpers 
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def vid2numpy(filename):
    """Reads a video and returns its contents in matrix form.

    Args:
        filename (pathlib.Path): a path to a video

    Returns:
        np.array(): matrix contents of the video 
    """
    vid = imageio.get_reader(str(filename), 'ffmpeg')
    # read all of video frames resulting in a (T, H, W, C) matrix 
    data = np.stack(list(vid.iter_data()))
    return data 

def images2numpy(filename):
    """Reads a fold of images and returns its contents in matrix form.

    Args:
        filename (pathlib.Path): a path to a folder of frames
        which make up a video. 

    Returns:
        np.array(): matrix contents of the video 
    """
    data = np.stack([plt.imread(frame_path) \
        for frame_path in filename.iterdir()])

    return data

# Decoding functions 
sequence_features = {
    'video_frames': tf.FixedLenSequenceFeature([], dtype=tf.string)
}

context_features = {
    'filename': tf.io.FixedLenFeature([], tf.string),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'temporal': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

def parse_example(example_proto):
    """Decodes a TFRecords example

    Args:
        example_proto (tf.train.Example): TFRecords Example

    Returns:
        tuple(tf.Tensor, int, str): tensor of the video, label and filename of 
        the video 
    """
    # Parse the input tf.train.Example using the dictionary above.
    context, sequence = tf.parse_single_sequence_example(example_proto,\
        context_features=context_features, sequence_features=sequence_features)
    # extract the expected shape 
    shape = (context['temporal'], context['height'], context['width'], context['depth'])

    ## the golden while loop ## 
    # loop through the feature lists and decode each image seperately:

    # decoding the first video 
    video_data = tf.image.decode_image(tf.gather(sequence['video_frames'], [0])[0])
    video_data = tf.expand_dims(video_data, 0)

    i = tf.constant(1, dtype=tf.int32)
    # condition of when to stop / loop through every frame
    cond = lambda i, _: tf.less(i, tf.cast(context['temporal'], tf.int32))

    # reading + decoding the i-th image frame 
    def body(i, video_data):
        # get the i-th index 
        encoded_img = tf.gather(sequence['video_frames'], [i])
        # decode the image 
        img_data = tf.image.decode_image(encoded_img[0]) 
        # append to list using tf operations 
        video_data = tf.concat([video_data, [img_data]], 0)
        # update counter & new video_data 
        return (tf.add(i, 1), video_data)

    # run the loop (use `shape_invariants` since video_data changes size)
    _, video_data = tf.while_loop(cond, body, [i, video_data], 
            shape_invariants=[i.get_shape(), tf.TensorShape([None])])
    # use this to set the shape + dtype
    video_data = tf.reshape(video_data, shape)
    video_data = tf.cast(video_data, tf.float32)

    label = context['label']
    filename = context['filename']

    return video_data, label, filename

