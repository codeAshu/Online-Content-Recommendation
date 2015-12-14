__author__ = 'Ashu'

import pandas as pd
import numpy as np
from summary import summary as SM
from utils import utils
from scipy.spatial.distance import cosine
from gensim import utils as genu
from gensim import corpora, models, similarities
import json
from uClassify import uClassify


path = "..\\data\\RecomEngine\\"
class Recomend:

    def getSummary(self, url):
        sm = SM()
        if url.find('pdf') > 0:
            d = sm.readPdf(url,25)
        elif url.find('youtube') > 0:
            d = sm.getYouTubeContext(url,25)
        else:
            d = sm.getSummaryFromUrl(url,25)
        return d

    def getNsimilarUsers(self,point, n):

        userVector = pd.io.pickle.read_pickle(path+'user_vector.pkl')
        X_train = userVector
        length = len(X_train)
        similarity = np.zeros(length)
        p1 =  point.values

        for index,row in X_train.iterrows():
            p2 = row.values
            similarity[index] = cosine(p1,p2)

        #find top n indexes
        ind = np.argpartition(similarity, -n)[-n:]

        #sort in descending order
        inds = ind[np.argsort(similarity[ind])][::-1]
        score = similarity[inds]
        users =[]
        for index, row in userVector.iterrows():
            if index in inds:
                users.append(row['user'])

        return users,score


    def querySummary(self,sf,url):
        """
        This will be replaced by database call to summary-master table
        """
        uu = utils()
        try:
            sumbyUrl = sf.groupby('url')
            ssm = sumbyUrl.get_group(url)['summary'].values[0]
            ssm = ''.join(s for s in ssm)
            ssm = genu.any2utf8(ssm)
            return sf, uu.cleanSummary(ssm)

        except KeyError:
            """
            when key error happen we can do as follows:
            either fetch the data at run time and insert in table
            or
            we can for now ignore those urls and send them to a job which
            will take the urls in a que and fetch & insrt their summary in a night job

            """

            print 'url summary not in table create one and insert in table also'
            newSum = ''.join(w for w in self.getSummary(url))
            newSum = genu.any2utf8( newSum)
            newSum =  uu.cleanSummary(newSum)

            #write program to insert into summary data frame
            sf2 = pd.DataFrame([[url,newSum]], columns=['url','summary'])
            sf = sf.append(sf2, ignore_index=True)
            sf.to_pickle(path+'summary.pkl')
            sf = pd.io.pickle.read_pickle(path+'summary.pkl')
            return sf, newSum

    """
    This function take list of recommended urls and user urls
    in one category and return a 2D matrix of similarity of there content
    """

    def getSimilarityMatrix(self,sf, recomUrls, userUrls):
        uu = utils()
        #create the corpus
        corpus = []
        for iindex,rowi in recomUrls.iterrows():
            ur1 = rowi['url']
            sf, sm1 = self.querySummary(sf,ur1)
            sm1 = genu.any2unicode(sm1)
            sm1 = uu.createTaggedDataForSummary(sm1)
            corpus.append(sm1)

        dictionary = corpora.Dictionary(corpus)
        corpusBow = [dictionary.doc2bow(text) for text in corpus]
        tfidf = models.TfidfModel(corpusBow)
        corpus_tfidf = tfidf[corpusBow]

        #create lsi model
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary)
        # corpus_lsi = lsi[corpus_tfidf]
        index = similarities.MatrixSimilarity(lsi[corpusBow])


        rCorpus = []
        for iindex,rowi in userUrls.iterrows():
            ur2 = rowi['url']
            sf, sm2 = self.querySummary(sf,ur2)
            sm2 = genu.any2unicode(sm2)
            sm2 = uu.createTaggedDataForSummary(sm2)
            rCorpus.append(sm2)

        # generate results
        vec_bow = [dictionary.doc2bow(text) for text in rCorpus]
        vec_lsi = lsi[vec_bow]
        sims = index[vec_lsi]

        """
        rows are user urls and columns are recommended urls
        """
        return  sims

    def getPossibleWithUserCF(self, sf,row,nUser):
        """
        This function is user-user collaborative filtering for finding
        similar users and get the possible recommendation from them
        """
        #target user
        tuser = row['user']

        #user  categories
        catByUser = pd.io.pickle.read_pickle(path+'categories.pkl').groupby('user')
        tprofile = catByUser.get_group(tuser)

        #row is a dataframe dictionalry
        tfeature = row


        #categorize url into its category
        urlCatByUser = pd.io.pickle.read_pickle(path+'url_category.pkl').groupby('user')

        turlCat = urlCatByUser.get_group(tuser)

        #get n similar useres and score
        users,score = self.getNsimilarUsers(tfeature,nUser)

        dl = []
        for u in users:
            dl.append(urlCatByUser.get_group(u))

        possibles = pd.concat(dl)

        return possibles

    def getRecommendationUrl(self,sf, row, n, threshold):

        #number of user for collaborative filtering
        n = 2
        urlCatByUser = pd.io.pickle.read_pickle(path+'url_category.pkl').groupby('user')
        #list of all categories
        cats = ['Arts','Business','Computers','Games','Health','Home','Recreation','Science','Society','Sports']

        #target user
        tuser = row['user']
        turlCat = urlCatByUser.get_group(tuser)

        #apply user-user collaborative filtering
        possibles = self.getPossibleWithUserCF(sf,row,n)

        recomTable = pd.io.pickle.read_pickle(path+'recommendation.pkl').groupby('url')


        """
        Now we have possible url matchs now categorize them in their category
        and fetch the summary of each url
        After that create a  simlilarity match of target user urls and possible a m*n matrix
        recommendation urls in each category and based on that suggest urls in each category
        """

        possiblebyCats = possibles.groupby('category')
        userbyCats = turlCat.groupby('category')

        recombyCat = []
        for catgry in cats:

            #recommended urls in this category
            try:
                rUrls = possiblebyCats.get_group(catgry)
                rUrls = rUrls.drop_duplicates('url')
            except KeyError:
                rUrls =[]

            #urls of user in this category
            try:
                usUrls = userbyCats.get_group(catgry)
                usUrls = usUrls.drop_duplicates('url')
            except KeyError:
                usUrls =[]

            rm = len(rUrls)
            un = len(usUrls)

            if(rm >0 and un >0):

                #now create a 2d matrix which will have summary matching among each url
                #matrix rows are userUrls and matrix columns are recommended url
                #and value is similarity score.

                mat = self.getSimilarityMatrix(sf, rUrls, usUrls)

                """
                Now find some of the highest matches in the matrix
                for those urls write a program to insert the recommendation
                into the table url-recommendation

                Later query that table first to search for recommendaitons

                """
                print 'recommendation urls are',rm
                print 'user urls are ', un

                print 'recommendations for category',catgry+" is "

                #find top indexes for each user url
                rowi = 0
                for row in mat:
                    forUrl = usUrls.values[rowi][1]
                    print 'recommendation for url ', forUrl +" are  follows :"
                    """
                    here create a tuple of category, url  and urls-recommended and return it
                    """
                    #select top n indices
                    ind = np.argpartition(row, -n)[-n:]
                    for i in ind:
                        score = row[i]
                        if(score> threshold ):
                            recomUrl =  rUrls.values[i][1]
                            print recomUrl
                            recombyCat.append({'Category':catgry,'url': forUrl,'recommendations': recomUrl })

                    rowi += 1
                    print '------------------------------------------'

        return sf, recombyCat

    def getCategorybyUrl(self,url, choice):

        read_api_key = "XX"
        write_api_key = "XX"

        a = uClassify()
        a.setApiKey(read_api_key)
        a.setOutputFormat("json")

        #set the URL to be classified
        a.setUrlToClassify(url)
        try:
            resp = a.classify(choice,"url")
            a= json.loads(resp)
            return a['cls1']
        except:
            return 'exception'

    def getNsimilarUrls(self, cat,urlCatDf,n,threshold):
        tableName = cat.lower()+'_url_category.pkl'
        print 'reading table '+tableName
        try:
            table = pd.io.pickle.read_pickle(path+tableName)

            recomByUrl = []
            length = len(table)
            for uIndex, uRow in urlCatDf.iterrows():
                url = uRow['url']
                a = table.groupby('url').get_group(url)
                columns = [column for column in table.columns if table[column].name != 'url']
                a = a[columns]
                p1 = a.values[0]

                similarity = np.zeros(length)
                for index,row in table.iterrows():

                    columns = [column for column in table.columns if table[column].name != 'url']
                    r = row[columns]
                    p2 =  r.values
                    cs = cosine(p1,p2)
                    similarity[index] = cs

                #check if similarity has atleast n items or not
                if len(similarity) <n:
                    n = len(similarity)

                #find top n indexes
                ind = np.argpartition(similarity, -n)[-n:]

                #sort in descending order
                inds = ind[np.argsort(similarity[ind])][::-1]

                for i in inds:
                    if similarity[i] > threshold:
                        recomUrl = table.irow(i)['url']
                        recomByUrl.append({'category':cat,  'url': url,'recommendations': recomUrl})

            return recomByUrl
        except :
            print 'could not find in table'

    """
    This is collaborative filtering with url
    It takes user urls from url_category table and do collaborative filtering
    Inputs :
    tUrlCat :  url_category row for a user
    n:         maximum number of recommendation against each url
    threshold : threshold score of matching of urls,only if matching cross the threshold ,it will be passed.
    """
    def getPossibleWithUrlCF(self, turlCat, n , threshold):
        """
        This is url - url collaborative filtering for finding similar urls
        """
        cats = ['Arts','Business','Computers','Games','Health','Home','Recreation','Science','Society','Sports']
        userUrlbyCats = turlCat.groupby('category')
        rList = []
        #now for each category
        for catgry in cats:
            try:
                usUrls = userUrlbyCats.get_group(catgry)
                usUrls = usUrls.drop_duplicates('url')
            except:
                usUrls =[]

            if len(usUrls) >0:
                #for each user url in this category find a similar url

                #load the category table
                resUrl = self.getNsimilarUrls(catgry, usUrls, n , threshold)
                if len(resUrl) >0:

                    #now check the score of each url-url match
                    for row in resUrl:
                        rList.extend(resUrl)
        return rList
