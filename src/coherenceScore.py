from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel


def cal_coherence(topic_dict, topics, index2word):
    data_set = [text.split() for texts_per_topic in topic_dict.values() for text in texts_per_topic]
    dictionary = corpora.Dictionary(data_set)

    for i in range(len(topics)):
        topics[i] = list(map(lambda x: dictionary.token2id[index2word[x]], topics[i][:20]))

    for T in [5, 10, 20]:
        cm_u_mass = CoherenceModel(topics=topics, texts=data_set, dictionary=dictionary, coherence='u_mass', topn=T)
        print("Coherence Score(T=%d) = %f" % (T, cm_u_mass.get_coherence()))

