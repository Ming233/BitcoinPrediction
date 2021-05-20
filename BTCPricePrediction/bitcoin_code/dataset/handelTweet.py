import numpy as np
import pandas as pd
import os
import sys
import argparse
import statistics
from sklearn.preprocessing import MinMaxScaler
import json
import csv
import torch
import torch.utils.data as Data

def main(): 
    files = ['2014-2015_tweetData.json', '2015-2016_tweetData.json', '2016-2017_tweetData.json', '2016-2017_tweetData.json','2017-2018_tweetData.json','2018-2019_tweetData.json','2019-2020_tweetData.json','2020-2021_tweetData.json']
    # Tweet
    vec = []
    sen = []
    date = []
    uid = []
    
    for file in files:
        with open (file) as openfileobject:
            data = json.load(openfileobject)
            index = 0
            data.reverse()
            for item in data:
                uid.append(item['_id']['$oid'])
                date.append(item['date']) 
                sen.append(item['Sentiment_Analysis'])
                vec.append(item['key_word_table'])
    
                index +=1
        
    #print(vec)   
    #print(sen)  
    
    alldate = []        
    
    with open('BTC_2014-2021.csv') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count !=0:
                if row['Date'] != '':
                    alldate.append(row['Date'])
            line_count+=1
        print(line_count)
                
    print('alldate')          
    print(alldate)
    
    outDatas = []
    
    date_count = 0
    date_not_exit = 0
    
    for i in range(len(alldate)):
        outData = {}
        print(alldate[i])
        if alldate[i] in date:
            ind = date.index(alldate[i])
            outData['date'] = date[ind]       
            outData['key_word_table'] = vec[ind]
            outData['Sentiment_Analysis'] = sen[ind]
            date_count += 1
            #print(1, outData)
            outDatas.append(outData)
        else:
            outData['date'] = alldate[i]   
            outData['key_word_table'] = [0]*11
            outData['Sentiment_Analysis'] = 0
            date_not_exit+=1
            #print(2, outData)
            outDatas.append(outData)
            
    print(date_count,' and ',date_not_exit)
    print(outDatas)
    name = 'tweet_2014-2021.json'
    
    with open(name, 'w') as outfile:
        json.dump(outDatas, outfile)        
    

main()