# Induction-of-sentiment-phrasal-lexicon-to-perform-Sentiment-Analysis
Induction of sentiment phrasal lexicon to perform Sentiment Analysis following the approach as described in the paper - Thumbs Up or Thumbs Down? Semantic Orientation Applied to Unsupervised Classification of Reviews. with the change as specified below:
Information Retrieval: instead of using an external search engine to collect hit counts, I wrote own search code to search in the training set, including implementing
the "NEAR" operator. Note that POS tags are needed to generate sentiment prhases. The tagged imbd corpus is provided. Each word has two tags, e.g., "thing NN I-NP", the first tag is the POS tag and the second tag is for chunking.
