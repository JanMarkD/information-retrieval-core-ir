import pyterrier as pt

def msmarco_generate():
    with pt.io.autoopen(dataset.get_corpus()[0], 'rt') as corpusfile:
        for l in corpusfile:
            docno, passage = l.split("\t")
            yield {'docno': docno, 'text': passage}


if __name__ == '__main__':
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

    dataset = pt.get_dataset("msmarco_passage")

    dataset_wiki = pt.get_dataset('irds:wikir/en78k')

    indexer_wiki = pt.IterDictIndexer('./wiki_index')
    index_ref_wiki = indexer_wiki.index(dataset_wiki.get_corpus_iter(), fields=['text'])

    print("Files in MSMARCO Passage Corpus: %s " % dataset.get_corpus())

    iter_indexer = pt.IterDictIndexer("./passage_index")

    indexref = iter_indexer.index(msmarco_generate(), meta={'docno': 20, 'text': 4096})

    indexref.toString()

    index = pt.IndexFactory.of("./passage_index/data.properties")
    index_wiki = pt.IndexFactory.of("./wiki_index/data.properties")

    bm25 = pt.BatchRetrieve.from_dataset(dataset, "terrier_stemmed", wmodel="BM25", verbose=True)
    bm25_wiki = pt.BatchRetrieve(index, wmodel="BM25", verbose=True)
    ifb2 = pt.BatchRetrieve.from_dataset(dataset, "terrier_stemmed", wmodel="IFB2", verbose=True)
    ifb2_wiki = pt.BatchRetrieve(index, wmodel="IFB2", verbose=True)

    improvement = (bm25_wiki % 100) >> pt.rewrite.RM3(index) >> bm25
    improvement_2 = (ifb2_wiki % 100) >> pt.rewrite.RM3(index) >> ifb2

    train_topics = dataset.get_topics('dev.small')
    test_topics = dataset.get_topics('test-2019')

    train_qrels = dataset.get_qrels('dev.small')

    test_qrels = dataset.get_qrels('test-2019')

    result = pt.Experiment(retr_systems=[bm25, ifb2, improvement, improvement_2],
                           topics=test_topics,
                           qrels=test_qrels,
                           eval_metrics=["map", "recip_rank"],
                           names=["BM25 Baseline", "IFB2", "BM25-RM3-WIKI-1", "IFB2-RM3-WIKI-1"],
                           save_dir="results",
                           verbose=True)

    print(result)
