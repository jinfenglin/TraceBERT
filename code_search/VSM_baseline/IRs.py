from gensim import corpora, models, matutils
from gensim.models import TfidfModel
from tqdm import tqdm


class VSM:
    def __init__(self):
        self.tfidf_model: TfidfModel = None

    def build_model(self, docs_tokens):
        print("Building VSM model...")
        dictionary = corpora.Dictionary(docs_tokens)
        corpus = [dictionary.doc2bow(x) for x in docs_tokens]
        self.tfidf_model = models.TfidfModel(corpus, id2word=dictionary)
        print("Finish building VSM model")

    def _get_doc_similarity(self, doc1_tk, doc2_tk):
        doc1_vec = self.tfidf_model[self.tfidf_model.id2word.doc2bow(doc1_tk)]
        doc2_vec = self.tfidf_model[self.tfidf_model.id2word.doc2bow(doc2_tk)]
        return matutils.cossim(doc1_vec, doc2_vec)

    def get_link_scores(self, source, target):
        s_tokens = source['tokens'].split()
        t_tokens = target['tokens'].split()
        score = self._get_doc_similarity(s_tokens, t_tokens)
        return score
