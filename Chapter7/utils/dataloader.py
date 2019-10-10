import os, re, io, glob, string, random
import torchtext
from torchtext.vocab import Vectors

def preprocessing_text(text):
    # 改行コードを消去
    text = re.sub('<br />', '', text)
    
    # カンマ，ピリオド以外の記号をスペースに置換
    for p in string.punctuation:
        if (p == ".") or (p ==","):
            # ピリオドとカンマの前後にはスペースを入れる
            text = text.replace(p, f" {p} ")
        else:
            text = text.replace(p, " ")
    
    return text

def tokenizer_punctuation(text):
    # スペースで単語分割を行う
    return text.strip().split()

def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    tokens = tokenizer_punctuation(text)
    return tokens

def get_IMDb_DataLoaders_and_TEXT(batch_size=24, max_length=256):
    data_dir = "../../datasets/ptca_datasets/chapter7"
    imdb_dir = os.path.join(data_dir, "aclImdb")
    cache_dir = os.path.join(data_dir, "vector_cache")
    train_tsv = "IMDb_train.tsv"
    test_tsv = "IMDb_test.tsv"
    english_fasttext_path = os.path.join(data_dir, "wiki-news-300d-1M.vec")

    TEXT = torchtext.data.Field(
        sequential=True,
        tokenize=tokenizer_with_preprocessing,
        use_vocab=True,
        lower=True, # 小文字化
        include_lengths=True,
        batch_first=True,
        fix_length=max_length,
        init_token="<cls>",
        eos_token="<eos>"
    )
    LABEL = torchtext.data.Field(
        sequential=False,
        use_vocab=False
    )

    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path=imdb_dir,
        train=train_tsv,
        test=test_tsv,
        format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)]
    )
    train_ds, val_ds = train_val_ds.split(
        split_ratio=0.8, 
        random_state=random.seed(1234)
    )

    english_fasttext_vectors = Vectors(
        name=english_fasttext_path,
        cache=cache_dir
    )
    
    TEXT.build_vocab(train_ds, vectors=english_fasttext_vectors, min_freq=10)
    train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True)
    val_dl = torchtext.data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)
    test_dl = torchtext.data.Iterator(test_ds, batch_size=batch_size, train=False, sort=False)
    
    return train_dl, val_dl, test_dl, TEXT
