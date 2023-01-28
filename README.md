# Induction-of-sentiment-phrasal-lexicon-to-perform-Sentiment-Analysis
Induction of sentiment phrasal lexicon to perform Sentiment Analysis following the approach as described in the paper - "Thumbs Up or Thumbs Down? Semantic Orientation Applied to Unsupervised Classification of Reviews" (Link: https://arxiv.org/ftp/cs/papers/0212/0212032.pdf). With the changes as specified below:
Information Retrieval: instead of using an external search engine to collect hit counts, I wrote own search code to search in the training set, including implementing
the "NEAR" operator. Note that POS tags are needed to generate sentiment prhases. The tagged imbd corpus is provided. Each word has two tags, e.g., "thing NN I-NP", the first tag is the POS tag and the second tag is for chunking. Details on how to run the code is provided in the report.

# Quanitfying gender biases in w2vNEWS embeddings as described in the paper: "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings"
Link: https://arxiv.org/pdf/1607.06520.pdf

Wrote code to identify the gender direction and to compute the direct bias of all the occupation words. Report shows the top biases for different occupations and how to run the code
