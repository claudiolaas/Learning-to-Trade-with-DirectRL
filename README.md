## Learning To Trade with DirectRL

This repo is an implementation of Teddy Kokers DirectRL approach to algorithmic trading.
- https://teddykoker.com/2019/06/trading-with-reinforcement-learning-in-python-part-ii-application/
- https://www.mdpi.com/1911-8074/13/8/178

I made the following changes from his setup in the jupyter notebook:
1. different returns function: although he does include a transaction cost rate, his formula does not capture the cumulative effect of fees. I think this can only be achieved in a loop. Please correct me if I'm wrong here.
2. automatic gradient calculation with pytorch (He mentions this in his paper).
3. optional use of a small NN as the trading function vs the linear function.
4. I use the sigmoid in my position function to avoid going short.

### Some findings
- I tried different lookback periods, features, NN architectures, normalization etc... None of it reliably outperformed buy and hold in the walk-forward-test :(
- using a NN as the trading function shows strong tendencies of overfitting (comparing in-sample vs out-of-sample returns)
- I tried many techniques of predicting the course of Bitcoin (not in this repo) and best I could do was as 0.52 accuracy of predicting next-step returns. I tried feeding this signal into this directRL system to take advantage of this edge but unfortunately it did not translate, although I did not explore this in depth.
