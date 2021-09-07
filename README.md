# RNN
Recurrent Neural Nrtwork on the WEATHERGOV dataset , Given a table of facts, we can generate a description suited for the given table using the basic Encoder-Decoder architecture.
# Please follow the below instruction for training.
```
1) Please set the path of save directory in run.sh before running it.
2) Number of epochs must be manually entered in code.
3) For beam search we have written beam.py
4) Attention mechanism is implemented in trainawsatt.py
5) BLEU score as early stopping criteria in file pcrnn.py
6) Run train.py for normal computation.
```

# To train the Encoder-Decoder model
```
python train.py --lr 0.001 --batch_size 100 --dropout_prob 0.9 --init 1 --save_dir "SAVE_DIR"
```

# Visualization of attention layer weights
In figures 1,2, and 3 color intensity represents how much attention was applied on that word during generation of summary. As we can see in all fles different words gets different attention to generate summary.
![image](https://user-images.githubusercontent.com/17472092/132383746-2e8b8352-20a4-4977-8c7c-3921ee7ef33f.png)
![image](https://user-images.githubusercontent.com/17472092/132383766-275f3c51-e642-4562-b27c-a03540cadb52.png)
![image](https://user-images.githubusercontent.com/17472092/132383788-c130c66c-2827-4de8-b00d-22b0cabe386d.png)
