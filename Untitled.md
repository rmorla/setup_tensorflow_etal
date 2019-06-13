

```python
!git clone https://github.com/calclavia/htm-tensorflow.git
```


```python
!pip install --user tqdm
```

    [33mThe directory '/.cache/pip/http' or its parent directory is not owned by the current user and the cache has been disabled. Please check the permissions and owner of that directory. If executing pip with sudo, you may want sudo's -H flag.[0m
    [33mThe directory '/.cache/pip' or its parent directory is not owned by the current user and caching wheels has been disabled. check the permissions and owner of that directory. If executing pip with sudo, you may want sudo's -H flag.[0m
    Collecting tqdm
    [?25l  Downloading https://files.pythonhosted.org/packages/45/af/685bf3ce889ea191f3b916557f5677cc95a5e87b2fa120d74b5dd6d049d0/tqdm-4.32.1-py2.py3-none-any.whl (49kB)
    [K    100% |████████████████████████████████| 51kB 1.3MB/s ta 0:00:011
    [?25hInstalling collected packages: tqdm
    [33m  The script tqdm is installed in '/.local/bin' which is not on PATH.
      Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.[0m
    Successfully installed tqdm-4.32.1
    [33mYou are using pip version 19.0.3, however version 19.1.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m



```python
!python mnist.py
```

    Using TensorFlow backend.
    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From /tf/code/htm/htm-tensorflow/layers/spatial_pooler.py:65: to_int64 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    WARNING:tensorflow:From /tf/code/htm/htm-tensorflow/layers/spatial_pooler.py:92: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Loading data...
    WARNING:tensorflow:From mnist.py:47: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please write your own downloading logic.
    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting data/train-images-idx3-ubyte.gz
    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting data/train-labels-idx1-ubyte.gz
    Extracting data/t10k-images-idx3-ubyte.gz
    Extracting data/t10k-labels-idx1-ubyte.gz
    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
    Processing data...
    100%|████████████████████████████████████| 55000/55000 [02:44<00:00, 334.83it/s]
    2019-06-11 10:39:13.925846: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
    2019-06-11 10:39:13.948368: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3696000000 Hz
    2019-06-11 10:39:13.949498: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x5f51b100 executing computations on platform Host. Devices:
    2019-06-11 10:39:13.949521: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
    === Epoch 0 ===
    Training HTM...
    100%|███████████████████████████████████████| 1546/1546 [00:39<00:00, 39.33it/s]
    Training classifier...
    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Epoch 1/10
    49500/49500 [==============================] - 1s 21us/step - loss: 0.5268 - acc: 0.8743
    Epoch 2/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.2450 - acc: 0.9301
    Epoch 3/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.1932 - acc: 0.9450
    Epoch 4/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.1643 - acc: 0.9528
    Epoch 5/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.1447 - acc: 0.9585
    Epoch 6/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.1300 - acc: 0.9632
    Epoch 7/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.1188 - acc: 0.9663
    Epoch 8/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.1094 - acc: 0.9699
    Epoch 9/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.1017 - acc: 0.9723
    Epoch 10/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.0948 - acc: 0.9746
    Validating...
    5500/5500 [==============================] - 0s 10us/step
    Accuracy: 0.9347272728139704
    === Epoch 1 ===
    Training HTM...
    100%|███████████████████████████████████████| 1546/1546 [00:37<00:00, 41.96it/s]
    Training classifier...
    Epoch 1/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.1809 - acc: 0.9418
    Epoch 2/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.1395 - acc: 0.9544
    Epoch 3/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.1196 - acc: 0.9614
    Epoch 4/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.1047 - acc: 0.9674
    Epoch 5/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.0933 - acc: 0.9715
    Epoch 6/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.0841 - acc: 0.9747
    Epoch 7/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.0764 - acc: 0.9780
    Epoch 8/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.0703 - acc: 0.9804
    Epoch 9/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.0645 - acc: 0.9827
    Epoch 10/10
    49500/49500 [==============================] - 1s 19us/step - loss: 0.0601 - acc: 0.9841
    Validating...
    5500/5500 [==============================] - 0s 8us/step
    Accuracy: 0.9519999998699535
    === Epoch 2 ===
    Training HTM...
     15%|██████                                  | 235/1546 [00:05<00:31, 41.97it/s]


```python

```
