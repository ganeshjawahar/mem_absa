## Aspect Level Sentiment Classification with Deep Memory Network

[TensorFlow](https://www.tensorflow.org/) implementation of [Tang et al.'s EMNLP 2016](https://arxiv.org/abs/1605.08900) work.

### Problem Statement
Given a sentence and an aspect occurring in the sentence, this task aims at inferring the sentiment polarity (e.g. positive, negative, neutral) of the aspect.

### Example
For example, in sentence ''great food but the service was dreadful!'', the sentiment polarity of aspect ''food'' is positive while the polarity of aspect ''service'' is negative.

### Quick Start
Download the 300-dimensional pre-trained word vectors from [Glove](http://nlp.stanford.edu/projects/glove/) and save it in the 'data' folder as 'data/glove.6B.300d.txt'. 

Train a model with 7 hops on the [Laptop](http://alt.qcri.org/semeval2016/task5/) dataset.
```
python main.py --show True
```

Note this code requires TensorFlow, Future and Progress packages to be installed. As of now, the model might not replicate the performance shown in the original paper as the authors have not yet confirmed the optimal hyper-parameters for training the memory network.

### Training options
* `edim`: internal state dimension [300]
* `lindim`: linear part of the state [75]
* `nhop`: number of hops [7]
* `batch_size`: batch size to use during training [128]
* `nepoch`: number of epoch to use during training [100]
* `init_lr`: initial learning rate [0.01]
* `init_hid`: initial internal state value [0.1]
* `init_std`: weight initialization std [0.05]
* `max_grad_norm`: clip gradients to this norm [50]
* `pretrain_file`: pre-trained glove vectors file path [../data/glove.6B.300d.txt]
* `train_data`: train gold data set path [./data/Laptop_Train_v2.xml] or [./data/Restaurants_Train_v2.xml]
* `test_data`: test gold data set path [./data/Laptops_Test_Gold.xml] or [./data/Restaurants_Test_Gold.xml]
* `show`: print progress [False]

### Performance - Laptop Dataset (todo)
| Model | In Paper | This Code|
|---|---|---|
|MemNet (1)|67.66||
|MemNet (2)|71.14||
|MemNet (3)|71.74||
|MemNet (4)|72.21||
|MemNet (5)|71.89||
|MemNet (6)|72.21||
|MemNet (7)|72.37||
|MemNet (8)|72.05||
|MemNet (9)|72.21||

### Performance - Restaurant Dataset (todo)
| Model | In Paper | This Code|
|---|---|---|
|MemNet (1)|76.10||
|MemNet (2)|78.61||
|MemNet (3)|79.06||
|MemNet (4)|79.87||
|MemNet (5)|80.14||
|MemNet (6)|80.05||
|MemNet (7)|80.32||
|MemNet (8)|80.14||
|MemNet (9)|80.95||

### Acknowledgements
* More than 80% of the code is borrowed from [carpedm20](https://github.com/carpedm20/MemN2N-tensorflow).
* Using this code means you have read and accepted the copyrights set by the dataset providers.

### Author
[Ganesh J](https://researchweb.iiit.ac.in/~ganesh.j/)

### Licence
MIT