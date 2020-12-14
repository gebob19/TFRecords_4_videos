# TFRecords_4_videos

Note: Package is still under construction

......... coming soon ...........

Associated blog post: [https://gebob19.github.io/tfrecords/](https://gebob19.github.io/tfrecords/)

| Method                      | Time       |
|-----------------------------|------------|
| Single Process              | 222+ hours |
| Multiprocess                | 11+ hours  |
| Multiprocess + Split (Ours) | 1 hour     |

Table1: Time to format the training set of something-something-v1

| Method                                | Time |
|---------------------------------------|------|
| Folder of Images                      | 38GB |
| TFRecord Encoding - 1MB shards (Ours) | 28GB |


Table2: Total size of something-something-v1 training set across formatting methods
