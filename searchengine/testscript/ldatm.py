#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling

__all__ = ['LDA']

import numpy
import time

numTopics = 20

class LDA:

    """
    A class used to perform Latent Dirichlet Allocation for learning topics
    ...

    Attributes
    ----------
    alpha : double
        parameter of topics prior
    beta : double
        parameter of words prior
    docs : list
        The documents in the corpus
    V : list
        The vocabulary
    z_m_n : double
        topics of words of documents
    n_m_z : integer
        word count of each document and topic
    n_m_z : integer
        word count of each topic and vocabulary
    n_z : integer
        word count of each topic
    N: integer
        The total number of words in the corpus

    Methods
    -------
    inference(self)
        learning once iteration

    worddist(self)
        get topic-word distribution
    
    perplexity(self, docs=None)
        Calculates the perplexity measure - inverse of the probability of the test set normalized by the number of words


    """

    def __init__(self, K, alpha, beta, docs, V, smartinit=True):
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.docs = docs
        self.V = V

        self.z_m_n = [] # topics of words of documents
        self.n_m_z = numpy.zeros((len(self.docs), K)) + alpha     # word count of each document and topic
        self.n_z_t = numpy.zeros((K, V)) + beta # word count of each topic and vocabulary
        self.n_z = numpy.zeros(K) + V * beta    # word count of each topic

        self.N = 0
        for m, doc in enumerate(docs):
            self.N += len(doc)
            z_n = []
            for t in doc:
                if smartinit:
                    p_z = self.n_z_t[:, t] * self.n_m_z[m] / self.n_z
                    z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                else:
                    z = numpy.random.randint(0, K)
                z_n.append(z)
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
            self.z_m_n.append(numpy.array(z_n))

    def inference(self):
        """Performs learning once iteration based on Collapsed Gibbs Sampling. Samples from a multinomial distribution or a uniform distribution as appropriate

        Parameters
        ----------
        None
.
        """
        for m, doc in enumerate(self.docs):
            z_n = self.z_m_n[m]
            n_m_z = self.n_m_z[m]
            for n, t in enumerate(doc):
                # discount for n-th word t with topic z
                z = z_n[n]
                n_m_z[z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                # sampling topic new_z for t
                p_z = self.n_z_t[:, t] * n_m_z / self.n_z
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                # set z the new topic and increment counters
                z_n[n] = new_z
                n_m_z[new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def worddist(self):
        """get topic-word distribution"""
        return self.n_z_t / self.n_z[:, numpy.newaxis]

    def perplexity(self, docs=None):
        """Calculates the perplexity measurewhich is inverse of the probability of the test set normalized by the number of words.
        This is just an exponentiation of the entropy and hence the learning iterations seek to minimize the perplexity in the 
        process of moving towards the right distribution of topics.


        Parameters
        ----------
        docs : list, optional
            The documents in the corpus

        """
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = self.n_m_z[m] / (len(self.docs[m]) + Kalpha)
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
            N += len(doc)
        return numpy.exp(log_per / N)



def lda_learning(lda, iteration, voca):
    pre_perp = lda.perplexity()
    lda_global = lda
    #for_comp = None
    count = 0
    print ("initial perplexity=%f" % pre_perp)
    for i in range(iteration):
        lda.inference()
        count+=1
        perp = lda.perplexity()
        print ("-%d p=%f" % (i + 1, perp))
        if pre_perp:
            if pre_perp < perp:
                # #output_word_topic_dist(lda, voca)
                # for_comp = pre_perp
                # print("yoyo", pre_perp)
                # time.sleep(10)
                # pre_perp = None
                print("hi")
                # time.sleep(5)
                # print("count is", count)
            else:
                lda_global = lda
                pre_perp = perp
    #if for_comp: print(for_comp)
    print(perp)
    #time.sleep(10)
    print("count is", count)
    print("---done---")
    output_word_topic_dist(lda_global, voca)
    #output_word_topic_dist(lda, voca)

def output_word_topic_dist(lda, voca):
    zcount = numpy.zeros(lda.K, dtype=int)
    wordcount = [dict() for k in range(lda.K)]
    for xlist, zlist in zip(lda.docs, lda.z_m_n):
        for x, z in zip(xlist, zlist):
            zcount[z] += 1
            if x in wordcount[z]:
                wordcount[z][x] += 1
            else:
                wordcount[z][x] = 1

    phi = lda.worddist()
    for k in range(lda.K):
        print ("\n-- topic: %d (%d words)" % (k, zcount[k]))
        for w in numpy.argsort(-phi[k])[:20]:
            print ("%s: %f (%d)" % (voca[w], phi[k,w], wordcount[k].get(w,0)))


def Convert(string): 
    li = list(string.split(" ")) 
    return li 

def main(q):
    start = time.time()
    from nltk.corpus import brown
    import optparse
    from . import vocabulary
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    #parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    our_range = "5:15"
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.5)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=5)

#CHANGED to 15 from 100 -- very important

    parser.add_option("-s", dest="smartinit", action="store_true", help="smart initialize of parameters", default=True)
    parser.add_option("--stopwords", dest="stopwords", help="exclude stop words", action="store_true", default=False)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int", help="threshold of document frequency to cut words", default=0)
    (options, args) = parser.parse_args()
    #if not (options.filename or options.corpus): parser.error("need corpus filename(-f) or corpus range(-c)")

    options.filename = 'D:\\corp.txt'
    if options.filename:
        corpus = vocabulary.load_file(options.filename)
    else:
        #corpus = vocabulary.load_corpus(options.corpus)
        corpus = vocabulary.load_corpus(our_range)
        if not corpus: parser.error("corpus range(-c) forms 'start:end'")
    if options.seed != None:
        numpy.random.seed(options.seed)

    voca = vocabulary.Vocabulary(options.stopwords)  #Instatiating an object of class Vocabulary
    docs = [voca.doc_to_ids(doc) for doc in corpus]  #Converting docs in corpus to corresponding ids using the methods of the voca object
    #print(docs.type)
    #print(docs.shape)
    print(type(docs))
    print(type(docs[0]))
    #time.sleep(5)  #SLEEPING FOR 5 SECONDS HERE
    if options.df > 0: docs = voca.cut_low_freq(docs, options.df)

    lda = LDA(options.K, options.alpha, options.beta, docs, voca.size(), options.smartinit)
    print ("corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(corpus), len(voca.vocas), options.K, options.alpha, options.beta))

    #import cProfile
    #cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
    lda_learning(lda, options.iteration, voca)
    end = time.time()
    print(end-start)
    text = "Indian Economy is the Best Economy"
    print(text)
    query = Convert(q)
    query_ids = voca.query_to_ids(query)

    print("Yo")
    print(query_ids)


    # global_maxi = -5
    # ans_doc = 1
    # for d, doc in enumerate(docs):
    #     total_sq = 0
    #     for w in query_ids:
    #         if w!=-1:
    #             maxi = -5
    #             for k in range(numTopics):
    #                 sqa = lda.n_z_t[k,w]*lda.n_z_t[k,w] + lda.n_m_z[d,k]*lda.n_m_z[d,k]
    #                 if sqa>maxi :
    #                     maxi = sqa
    #             total_sq+=maxi
    #     if total_sq>global_maxi and total_sq!=0:
    #         global_maxi = total_sq
    #         ans_doc = d
    # print("Yass", ans_doc)

#Calculating the similarity

    score_doc = {}
    score = []
    ans_doc = 1
    for d, doc in enumerate(docs):
        total_sq = 0
        for w in query_ids:
            if w!=-1:
                maxi = -5
                for k in range(numTopics):
                    sqa = lda.n_z_t[k,w]*lda.n_z_t[k,w] + lda.n_m_z[d,k]*lda.n_m_z[d,k]
                    if sqa>maxi :
                        maxi = sqa
                total_sq+=maxi
        score_doc[total_sq] = d
        score.append(total_sq)
    
    ans = []    
    score.sort(reverse = True)
    
    #for i in range(len(score)):
     #   print(score_doc[score[i]])

    for i in range(len(score)):
        #ans_doc = score_doc[score[i]]
        if(score_doc[score[i]] not in ans): 
            ans.append(score_doc[score[i]])
        #print(score_doc[score[i]])
        #print("\n Yasss", corpus[ans_doc])
    print(ans)
    return(ans)    

    
    #ans = []
    #print(corpus[ans_doc])
    #ans.append(corpus[ans_doc])
    # ans.append(ans_doc)
    # ans.append(345)
    # ans.append(3982)
    # ans.append(8194)
    # ans.append(1920)
    #return(ans)
    #print(corpus[10])
    #print(corpus)


    #query = the text string as a list

"""

    #Implement vectorizer
    #Get the same lda model
    #Implement transform

    data = []
 
    for fileid in brown.fileids():
        document = ' '.join(brown.words(fileid))
        data.append(document)

    data = data[5:15]



    text = "The economy is working better than ever"

    from sklearn.feature_extraction.text import CountVectorizer
 
    #NUM_TOPICS = 10
 
    vectorizer = CountVectorizer(min_df=5, max_df=0.9, 
                             stop_words='english', lowercase=True, 
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    
    data_vectorized = vectorizer.fit_transform(data)

    m = vectorizer.transform([text])
    print(m[0])



"""

    #lda.predict(m, lda)


    #x = lda_model.transform(m)[0]


    #x = lda_model.transform(vectorizer.transform([text]))[0]
    #print(x)

    #from sklearn.metrics.pairwise import euclidean_distances
 
    # def most_similar(x, Z, top_n=5):
    #     dists = euclidean_distances(x.reshape(1, -1), Z)
    #     pairs = enumerate(dists[0])
    #     most_similar = sorted(pairs, key=lambda item: item[1])[:top_n]
    #     return most_similar
    
    # #similarities = most_similar(x, nmf_Z)
    # #similarities = most_similar(x, lda_Z)
    # similarities = most_similar(x, lda)
    # document_id, similarity = similarities[0]
    # print(data[document_id][:1000])

if __name__ == "__main__":
    main()