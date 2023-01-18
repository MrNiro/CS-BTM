from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel


def cal_coherence(topic_dict, topics, index2word):
    dataset = [text.split() for texts_per_topic in topic_dict.values() for text in texts_per_topic]
    dictionary = corpora.Dictionary(dataset)
    for i in range(len(topics)):
        topics[i] = list(map(lambda x: dictionary.token2id[index2word[x]], topics[i][:20]))

    cal_coherence_(dataset, topics, dictionary)


def cal_coherence_(dataset, topic_words, dictionary=None):
    if not dictionary:
        dictionary = corpora.Dictionary(dataset)
    for T in [5, 10, 20]:
        cm_u_mass = CoherenceModel(topics=topic_words, texts=dataset,
                                   dictionary=dictionary, coherence='u_mass', topn=T)
        print("Coherence Score(T=%d) = %f" % (T, cm_u_mass.get_coherence()))
