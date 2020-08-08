import gensim
import numpy
from gensim import corpora, models, matutils
from gensim.models import TfidfModel


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


class LDA:
    def __init__(self):
        self.ldamodel = None

    def build_model(self, docs_tokens, num_topics=200, passes=1000):
        dictionary = corpora.Dictionary(docs_tokens)
        corpus = [dictionary.doc2bow(x) for x in docs_tokens]
        self.ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary,
                                                        passes=passes, alpha='auto',
                                                        random_state=numpy.random.RandomState(1))

    def get_topic_distrb(self, doc):
        bow_doc = self.ldamodel.id2word.doc2bow(doc)
        return self.ldamodel.get_document_topics(bow_doc)

    def get_link_scores(self, source, target):
        """
        :param doc1_tk: Preprocessed documents as tokens
        :param doc2_tk: Preprocessed documents as tokens
        :return:
        """
        doc1_tk = source['tokens'].split()
        doc2_tk = target['tokens'].split()
        dis1 = self.get_topic_distrb(doc1_tk)
        dis2 = self.get_topic_distrb(doc2_tk)
        # return 1 - matutils.hellinger(dis1, dis2)
        return matutils.cossim(dis1, dis2)


class LSI:
    def __init__(self):
        self.lsi = None

    def build_model(self, docs_tokens, num_topics=200):
        dictionary = corpora.Dictionary(docs_tokens)
        corpus = [dictionary.doc2bow(x) for x in docs_tokens]
        self.lsi = gensim.models.LsiModel(corpus, num_topics=num_topics, id2word=dictionary)

    def get_topic_distrb(self, doc):
        bow_doc = self.lsi.id2word.doc2bow(doc)
        return self.lsi[bow_doc]

    def get_link_scores(self,  source, target):
        doc1_tk = source['tokens'].split()
        doc2_tk = target['tokens'].split()
        dis1 = self.get_topic_distrb(doc1_tk)
        dis2 = self.get_topic_distrb(doc2_tk)
        return matutils.cossim(dis1, dis2)
