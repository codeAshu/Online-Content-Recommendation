__author__ = 'Ashu'


import pandas as pd
import numpy as np
from summary import summary
from utils import utils
from scipy.spatial.distance import cosine
from gensim import utils as genu
from gensim import corpora, models, similarities
import json
from uClassify import uClassify
import re
from collections import Counter

"""
Complete flow:

So a user profile comes


1. write a batch import function for user which will insert it into summary master,
also drops the dead links.

2. once that import is done, from summary do the api call to get the category for
that url and insert into category master together with userid
3. create a user profile table too with minimal field

4. Now create userVector and insert into table

5. assign uservector row as X_test = userVector[userId]  for the user Id

6. After that run the above programm by providing

"""
from Engine import path

class batchProcess:


    def batchImport(self,profile):
        urls = profile['urls']
        userid = profile['userid']
        info = {i:profile[i] for i in profile if i!='urls'}

        #insert into profile_master
        print 'inserting user profile...'
        self.insertUserProfile(info)

        #insert into Summary
        print 'inserting summary ...'
        sumList = self.insertSummary(urls)


        #now calculate category of the url by the summary
        print 'inserting categories...'
        self.insertCategoris(userid, sumList)

        #now create feature vector
        print 'creating user vector ...'
        self.createUserVector(userid)

        #create url_category
        print 'create url categories ...'
        self.createUrlCategory(userid)

        #insert micro category
        print 'create micro categories...'
        self.insertMicroCategory(sumList)


        print 'batch import of profile is done !!'

    """
    This fnction pick the highest category for an url from dfbyUser database
    """
    def K(self,x):

        url = x['url']
        user =  x['user']
        cats =  np.array(x.values[:-2])
        #find top n indexes
        ind = np.argpartition(cats, -1)[-1:][0]

        score =  cats[ind]
        category=  x.keys()[ind]

        data = [user,url,category,score]
        return data

    def createUrlCategory(self,userid):
        category = pd.io.pickle.read_pickle(path+'categories.pkl')
        categoryByUser = category.groupby('user').get_group(userid)
        url_catDf = pd.io.pickle.read_pickle(path+'url_category.pkl')

        urlCat = []
        a = categoryByUser.T.apply(self.K)
        for row in a:
            urlCat.append(row)
        cols = ['user','url','category','score']
        urlCat = pd.DataFrame(urlCat, columns=cols)

        ndf  = url_catDf.append(urlCat, ignore_index=True)
        ndf.to_pickle(path+'url_category.pkl')

    """
    This should be after url_category
    """
    def insertMicroCategory(self,sumList):

        urlCat = pd.io.pickle.read_pickle(path+'url_category.pkl').groupby('url')
        for url,summary in sumList:
            if len(summary)>1600:
                 summary = summary[:1600]

            try:

                cat = urlCat.get_group(url)['category'].values[0].lower()

                #load table
                tName = cat+'_url_category.pkl'
                catTable = pd.io.pickle.read_pickle(path+tName)

                #get category
                d = self.getCategorybyText(summary,cat)

                #insert into table
                if d != 'exception':
                    df = pd.DataFrame([d])
                    df['url'] = url
                    catTable = catTable.append(df, ignore_index=True)
            except:
                print 'did not find in url_cat'
            catTable.to_pickle(path+tName)
            print 'micro cat insert in ', cat +' is done.'


    """
    create user vector for each user
    """
    def createUserVector(self,userid):
        category = pd.io.pickle.read_pickle(path+'categories.pkl')
        categoryByUser = category.groupby('user').get_group(userid)

        mean = categoryByUser[['Arts','Business','Computers','Games','Health','Home','Recreation','Science','Society','Sports']].mean()
        mean['user'] = userid
        userVector  = pd.io.pickle.read_pickle(path+'user_vector.pkl')
        k = userVector.append(mean,ignore_index=True)
        k.to_pickle(path+'user_vector.pkl')


    """
    if both has same highest -- good go for avg
    if both has different highest:
        1. if both have above 0.95 -- discard (disagree) ( from summary table  on some  anomaly flag for it) return exception
        2. if one has 0.95 and above other has below 0.7 -- go for the first
        3. rest all take avg
    """
    def determineCategory(self, c1,c2):

        c1_maxv = -1
        c1_maxk = ''

        for p in c1:
            value =  c1[p]
            if value > c1_maxv:
                c1_maxv =value
                c1_maxk = p

        c2_maxv = -1
        c2_maxk = ''

        for p in c2:
            value =  c2[p]
            if value > c2_maxv:
                c2_maxv =value
                c2_maxk = p

        #now we have maximum value and maximum key for both c1 and c2
        if c1_maxv > 0.98 and c2_maxv > 0.98 :
            return 'exception'

        elif (c1_maxv > 0.98 and c2_maxv <0.7) or (c2_maxv > 0.98 and c1_maxv <0.7):
            if c1_maxv > c2_maxv:
                return c1
            else:
                return c2
        else:
            c = dict( Counter(c1)+Counter(c2) )
            d2 = {k: v/2.0 for k, v in c.items()}
            return d2


    """
    create user vector for each user
    """
    def insertCategoris(self, userid,sumList):
        #sumList is list of list of type [url,summary]
        old = pd.io.pickle.read_pickle(path+'categories.pkl')
        catList = []
        for url,summary in sumList:
            if len(summary)>1600:
                summary = summary[:1600]
            #both should agree on highest category -- otherwise cancel it
            catA = self.getCategorybyText(summary,'topic')
            catB = self.getCategorybyUrl(url,'topic')
            if catA != 'exception' and catB != 'exception':
                category = self.determineCategory(catA, catB)
                if category != 'exception':
                    category['url'] =  url
                    category['user'] = userid
                    catList.append(category)
        new = pd.DataFrame(catList)
        catDf = old.append(new, ignore_index=True)
        catDf.to_pickle(path+'categories.pkl')


    """
    insert user profile info in profile-master
    """
    def insertUserProfile(self,row):

        print row
        old = pd.io.pickle.read_pickle(path+'profile_master.pkl')
        new = pd.DataFrame([row])
        profile  = old.append(new, ignore_index=True)
        profile.to_pickle(path+'profile_master.pkl')





    def urlify(self,s):
        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", '', s)

        # Replace all runs of whitespace with a single dash
        s = re.sub(r"\s+", '+', s)

        return s

    def getCategorybyText(self,text, choice):

        read_api_key = "mcSJ8caveJDds94kMdjgeV7QVFE"
        write_api_key = "OxyyuUSOfgWGdHSggGJwETPWA"

        text = ''.join(w for w in text)
        text = self.urlify(text)

        a = uClassify()
        a.setApiKey(read_api_key)
        a.setOutputFormat("json")


        #set the URL to be classified
        a.setTextToClassify(text)
        try:
            resp = a.classify(choice,"text")
            a= json.loads(resp)
            cats = a['cls1']
            return cats
        except:
            return 'exception'

    def getSummary(self, url):
        sm = summary()
        if url.find('pdf') > 0:
            d = sm.readPdf(url,15)
        elif url.find('youtube') > 0:
            d = sm.getYouTubeContext(url,15)
        else:
            d = sm.getSummaryFromUrl(url,15)
        return d

    def insertSummary(self,urls):
        old = pd.io.pickle.read_pickle(path+'summary.pkl')
        sumByUrl = old.groupby('url')
        uu = utils()

        sumList = []
        for url in urls:
            try:
                ssm = sumByUrl.get_group(url)['summary'].values[0]
                sumList.append([url,ssm])
            except KeyError:
                newSum = ''.join(w for w in self.getSummary(url))
                newSum = genu.any2utf8( newSum)
                newSum =  uu.cleanSummary(newSum)
                if len(newSum)<100:
                    print 'dropping'+newSum +' of url '+url
                else:
                    print url +' summary not in table create one and insert in table also'
                    sumList.append([url,newSum])
        if len(sumList) > 0:
            sf = pd.DataFrame(sumList, columns=['url','summary'])
            sumDf = old.append(sf, ignore_index=True)
            sumDf.to_pickle(path+'summary.pkl')
        return sumList

    def getCategorybyUrl(self,url, choice):

        read_api_key = "mcSJ8caveJDds94kMdjgeV7QVFE"
        write_api_key = "OxyyuUSOfgWGdHSggGJwETPWA"

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