# Batch Incremental Gaussian Process Regression - BIGPR

Incremental Gaussian Process Regression is a known method to avoid recompute the kernel each time.
The forked implementation missed many optimizations opportunities (as I'm sure mine will, too).

Here we add support for batchly update the kernel matrix, and also a couple algorithms to avoid recompute the inverse, that gain much time.

---

Refer to this book: Rasmussen, Carl Edward. "Gaussian processes in machine learning." Summer School on Machine Learning. Springer, Berlin, Heidelberg, 2003.   

Refer to this paper: Nguyen–Tuong, Duy, and Jan Peters. "Incremental sparsification for real-time online model learning." Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics. 2010.
