__author__ = 'Ashu'

import csv
from BatchImport import  batchProcess as bp
from DailyUpdate import DailyProcess as dp
import pandas as pd
from Engine import Recomend
bpo = bp()
rm = Recomend()
dpo =dp()
from Engine import path


"""
Import batch Data here
"""

def testBatchImport(userid, name,urls):


    userProfile = {'userid':userid, 'userName': name,'location':'Bangalore',
                   'urls':urls }

    bpo.batchImport(userProfile)


"""
Daily update
"""

def testDailyUpdate(userid,urls):

   profile = {'userid':userid, 'urls':urls}
   dpo.addUrls(profile)


"""
Testing here
"""
def testUserCFRecommendation(userid):

    userVector = pd.io.pickle.read_pickle(path+'user_vector.pkl')
    X_test = userVector.groupby('user').get_group(userid)

    sf = pd.io.pickle.read_pickle(path+'summary.pkl')
    for index, row in X_test.iterrows():
        sf, recombyCat = rm.getRecommendationUrl(sf,row, 2,0.7)
        sf.to_pickle(path+'summary.pkl')
        sf = pd.io.pickle.read_pickle(path+'summary.pkl')
        old =  pd.io.pickle.read_pickle(path+'recommendation.pkl')
        recommendation = pd.DataFrame(recombyCat)
        recommendation['score']  = 0
        nr = old.append(recommendation, ignore_index=True)
        nr.drop_duplicates()
        nr.to_pickle(path+'recommendation.pkl')
        return recombyCat

def testUrlCFRecommendation(userid):
    turlCat = pd.io.pickle.read_pickle(path+'url_category.pkl').groupby('user').get_group(userid)
    return rm.getPossibleWithUrlCF(turlCat,3,0.7)

# if  __name__ =='__main__':
#     print testUrlCFRecommendation(50)
#
#     userid = 55
#     name = 'name55'
#     testBatchImport(userid,name)