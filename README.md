# tensor_a
"""Functions for downloading and reading MNIST data.""" 
17 from __future__ import absolute_import 
18 from __future__ import division 
19 from __future__ import print_function 
20  
21 import gzip 
22 import os 
23  
24 import tensorflow.python.platform 
25  
26 import numpy 
27 from six.moves import urllib 
28 from six.moves import xrange  # pylint: disable=redefined-builtin 
29 import tensorflow as tf 
30  
31 SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/' 
32  
33  
34 def maybe_download(filename, work_directory): 
35   """Download the data from Yann's website, unless it's already here.""" 
36   if not os.path.exists(work_directory): 
37     os.mkdir(work_directory) 
38   filepath = os.path.join(work_directory, filename) 
39   if not os.path.exists(filepath): 
40     filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath) 
41     statinfo = os.stat(filepath) 
42     print('Successfully downloaded', filename, statinfo.st_size, 'bytes.') 
43   return filepath 
44  
45  
46 def _read32(bytestream): 
47   dt = numpy.dtype(numpy.uint32).newbyteorder('>') 
48   return numpy.frombuffer(bytestream.read(4), dtype=dt)[0] 
49  
50  
51 def extract_images(filename): 
52   """Extract the images into a 4D uint8 numpy array [index, y, x, depth].""" 
53   print('Extracting', filename) 
54   with gzip.open(filename) as bytestream: 
55     magic = _read32(bytestream) 
56     if magic != 2051: 
57       raise ValueError( 
58           'Invalid magic number %d in MNIST image file: %s' % 
59           (magic, filename)) 
60     num_images = _read32(bytestream) 
61     rows = _read32(bytestream) 
62     cols = _read32(bytestream) 
63     buf = bytestream.read(rows * cols * num_images) 
64     data = numpy.frombuffer(buf, dtype=numpy.uint8) 
65     data = data.reshape(num_images, rows, cols, 1) 
66     return data 
67  
68  
69 def dense_to_one_hot(labels_dense, num_classes=10): 
70   """Convert class labels from scalars to one-hot vectors.""" 
71   num_labels = labels_dense.shape[0] 
72   index_offset = numpy.arange(num_labels) * num_classes 
73   labels_one_hot = numpy.zeros((num_labels, num_classes)) 
74   labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1 
75   return labels_one_hot 
76  
77  
78 def extract_labels(filename, one_hot=False): 
79   """Extract the labels into a 1D uint8 numpy array [index].""" 
80   print('Extracting', filename) 
81   with gzip.open(filename) as bytestream: 
82     magic = _read32(bytestream) 
83     if magic != 2049: 
84       raise ValueError( 
85           'Invalid magic number %d in MNIST label file: %s' % 
86           (magic, filename)) 
87     num_items = _read32(bytestream) 
88     buf = bytestream.read(num_items) 
89     labels = numpy.frombuffer(buf, dtype=numpy.uint8) 
90     if one_hot: 
91       return dense_to_one_hot(labels) 
92     return labels 
93  
94  
95 class DataSet(object): 
96  
97   def __init__(self, images, labels, fake_data=False, one_hot=False, 
98                dtype=tf.float32): 
99     """Construct a DataSet. 
100  
101     one_hot arg is used only if fake_data is true.  `dtype` can be either 
102     `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into 
103     `[0, 1]`. 
104     """ 
105     dtype = tf.as_dtype(dtype).base_dtype 
106     if dtype not in (tf.uint8, tf.float32): 
107       raise TypeError('Invalid image dtype %r, expected uint8 or float32' % 
108                       dtype) 
109     if fake_data: 
110       self._num_examples = 10000 
111       self.one_hot = one_hot 
112     else: 
113       assert images.shape[0] == labels.shape[0], ( 
114           'images.shape: %s labels.shape: %s' % (images.shape, 
115                                                  labels.shape)) 
116       self._num_examples = images.shape[0] 
117  
118       # Convert shape from [num examples, rows, columns, depth] 
119       # to [num examples, rows*columns] (assuming depth == 1) 
120       assert images.shape[3] == 1 
121       images = images.reshape(images.shape[0], 
122                               images.shape[1] * images.shape[2]) 
123       if dtype == tf.float32: 
124         # Convert from [0, 255] -> [0.0, 1.0]. 
125         images = images.astype(numpy.float32) 
126         images = numpy.multiply(images, 1.0 / 255.0) 
127     self._images = images 
128     self._labels = labels 
129     self._epochs_completed = 0 
130     self._index_in_epoch = 0 
131  
132   @property 
133   def images(self): 
134     return self._images 
135  
136   @property 
137   def labels(self): 
138     return self._labels 
139  
140   @property 
141   def num_examples(self): 
142     return self._num_examples 
143  
144   @property 
145   def epochs_completed(self): 
146     return self._epochs_completed 
147  
148   def next_batch(self, batch_size, fake_data=False): 
149     """Return the next `batch_size` examples from this data set.""" 
150     if fake_data: 
151       fake_image = [1] * 784 
152       if self.one_hot: 
153         fake_label = [1] + [0] * 9 
154       else: 
155         fake_label = 0 
156       return [fake_image for _ in xrange(batch_size)], [ 
157           fake_label for _ in xrange(batch_size)] 
158     start = self._index_in_epoch 
159     self._index_in_epoch += batch_size 
160     if self._index_in_epoch > self._num_examples: 
161       # Finished epoch 
162       self._epochs_completed += 1 
163       # Shuffle the data 
164       perm = numpy.arange(self._num_examples) 
165       numpy.random.shuffle(perm) 
166       self._images = self._images[perm] 
167       self._labels = self._labels[perm] 
168       # Start next epoch 
169       start = 0 
170       self._index_in_epoch = batch_size 
171       assert batch_size <= self._num_examples 
172     end = self._index_in_epoch 
173     return self._images[start:end], self._labels[start:end] 
174  
175  
176 def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=tf.float32): 
177   class DataSets(object): 
178     pass 
179   data_sets = DataSets() 
180  
181   if fake_data: 
182     def fake(): 
183       return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype) 
184     data_sets.train = fake() 
185     data_sets.validation = fake() 
186     data_sets.test = fake() 
187     return data_sets 
188  
189   TRAIN_IMAGES = 'train-images-idx3-ubyte.gz' 
190   TRAIN_LABELS = 'train-labels-idx1-ubyte.gz' 
191   TEST_IMAGES = 't10k-images-idx3-ubyte.gz' 
192   TEST_LABELS = 't10k-labels-idx1-ubyte.gz' 
193   VALIDATION_SIZE = 5000 
194  
195   local_file = maybe_download(TRAIN_IMAGES, train_dir) 
196   train_images = extract_images(local_file) 
197  
198   local_file = maybe_download(TRAIN_LABELS, train_dir) 
199   train_labels = extract_labels(local_file, one_hot=one_hot) 
200  
201   local_file = maybe_download(TEST_IMAGES, train_dir) 
202   test_images = extract_images(local_file) 
203  
204   local_file = maybe_download(TEST_LABELS, train_dir) 
205   test_labels = extract_labels(local_file, one_hot=one_hot) 
206  
207   validation_images = train_images[:VALIDATION_SIZE] 
208   validation_labels = train_labels[:VALIDATION_SIZE] 
209   train_images = train_images[VALIDATION_SIZE:] 
210   train_labels = train_labels[VALIDATION_SIZE:] 
211  
212   data_sets.train = DataSet(train_images, train_labels, dtype=dtype) 
213   data_sets.validation = DataSet(validation_images, validation_labels, 
214                                  dtype=dtype) 
215   data_sets.test = DataSet(test_images, test_labels, dtype=dtype) 
216  
217   return data_sets 
