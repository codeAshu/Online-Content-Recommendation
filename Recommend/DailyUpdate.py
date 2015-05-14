__author__ = 'Ashu'


import pandas as pd
import numpy as np
from BatchImport import batchProcess as bp
"""
This class work on url addition to the existing user

Query the userProfile table tocheck user exist

Now check if the url exist in summary table -- if not add it

check if url is added in the categorization also, if yes thn stop

if not insert into each table

"""
from Engine import path

class DailyProcess:

    def addUrls(self, profile):
        '''user profile would be dict of userid and list of urls'''
        userid = profile['userid']
        urlList = profile['urls']

        #load tables
        pTable = pd.io.pickle.read_pickle(path+'profile_master.pkl')

        try:
            userProfile = pTable.groupby('userid').get_group(userid)
            print 'user found updating its info ....'
        except KeyError:
            print "user does not exist please batch Import the user information"
            return

        #create batch process object
        bpo = bp()
        urlSumList = bpo.insertSummary(urlList)
        bpo.insertCategoris(userid, urlSumList)
        self.updateUserVector(userid)
        self.updateUrlCategory(userid)

        #insert micro category
        bpo.insertMicroCategory(urlSumList)


        print  'follwoing usrls added to list'
        for url,s in urlSumList:
            print url


    """
    create user vector for each user
    """
    def updateUserVector(self,userid):
        category = pd.io.pickle.read_pickle(path+'categories.pkl')
        categoryByUser = category.groupby('user').get_group(userid)

        mean = categoryByUser[['Arts','Business','Computers','Games','Health','Home','Recreation','Science','Society','Sports']].mean()
        mean['user'] = userid
        userVector  = pd.io.pickle.read_pickle(path+'user_vector.pkl')
        userVector = userVector[userVector.user != userid]
        k = userVector.append(mean,ignore_index=True)
        k.to_pickle(path+'user_vector.pkl')

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

    def updateUrlCategory(self,userid):
        category = pd.io.pickle.read_pickle(path+'categories.pkl')
        categoryByUser = category.groupby('user').get_group(userid)
        url_catDf = pd.io.pickle.read_pickle(path+'url_category.pkl')
        url_catDf = url_catDf[url_catDf.user !=userid]

        urlCat = []
        a = categoryByUser.T.apply(self.K)
        for row in a:
            urlCat.append(row)
        cols = ['user','url','category','score']
        urlCat = pd.DataFrame(urlCat, columns=cols)

        ndf  = url_catDf.append(urlCat, ignore_index=True)
        ndf.to_pickle(path+'url_category.pkl')