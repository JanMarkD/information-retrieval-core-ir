import pyterrier as pt
from sklearn.ensemble import RandomForestRegressor

def msmarco_generate():
    with pt.io.autoopen(dataset.get_corpus()[0], 'rt') as corpusfile:
        for l in corpusfile:
            docno, passage = l.split("\t")
            yield {'docno': docno, 'text': passage}


if __name__ == '__main__':
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

    dataset = pt.get_dataset("msmarco_passage")

    print("Files in MSMARCO Passage Corpus: %s " % dataset.get_corpus())

    iter_indexer = pt.IterDictIndexer("./passage_index")

    indexref = iter_indexer.index(msmarco_generate(), meta={'docno': 20, 'text': 4096})

    indexref.toString()

    index = pt.IndexFactory.of("./passage_index/data.properties")
    print(index.getCollectionStatistics().toString())

    bm25 = pt.BatchRetrieve.from_dataset(dataset, "terrier_stemmed", wmodel="BM25", verbose=True)
    tf = pt.BatchRetrieve.from_dataset(dataset, "terrier_stemmed", wmodel="Tf", verbose=True)
    pl2 = pt.BatchRetrieve.from_dataset(dataset, "terrier_stemmed", wmodel="PL2", verbose=True)

    pipeline = (bm25 % 100) >> (tf ** pl2)

    rf = RandomForestRegressor(n_estimators=400, verbose=2)
    final_pipe = pipeline >> pt.ltr.apply_learned_model(rf)

    train_topics = dataset.get_topics('train')
    train_qrels = dataset.get_qrels('train')

    test_topics = dataset.get_topics('test-2019')
    test_qrels = dataset.get_qrels('test-2019')

    final_pipe.fit(train_topics, train_qrels)

    result = pt.Experiment(retr_systems=[bm25, final_pipe],
                           topics=test_topics,
                           qrels=test_qrels,
                           eval_metrics=["map"],
                           names=["BM25 Baseline", "LTR"],
                           verbose=True)

    print(result)
