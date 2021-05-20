import nltk
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re, string
from nltk.corpus import stopwords
from nltk import FreqDist
import json
import pymongo
import sys
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence
 
def remove_noise(tweet_tokens, stop_words = ()):
 
    cleaned_tokens = []
 
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
 
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
 
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
 
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token
            
if __name__ == "__main__":
    #file_name = '2014-2015_data.json'
    file = open('frquency.txt', 'a', encoding="utf-8")
    file_name_list = ['2014-2015_data.json','2015-2016_data.json','2016-2017_data.json','2017-2018_data.json','2018-2019_data.json','2019-2020_data.json','2020-2021_data.json']
    for file_name in file_name_list:
        stop_words = stopwords.words('english')
        tweets = twitter_samples.strings(file_name)
        tweet_tokens = twitter_samples.tokenized(file_name)     
        analyzer = SentimentIntensityAnalyzer()
        
        print("Start token clean!")
        cleaned_tokens_list = []
        for tokens in tweet_tokens:
            lemmatize_token = lemmatize_sentence(tokens)
            cleaned_tokens_list.append(remove_noise(lemmatize_token, stop_words))
        print("cleaned_tokens_list Done!")
        
        #creating one_hot vector, if dont need please commont
        #======================================
        all_key_word_list = []
        key_word_list = ['btc','buy','go','people','year','time','like','...','money','price','crypto']
        for i in range(len(cleaned_tokens_list)):
            #print(cleaned_tokens_list[i])
            key_word_table = [0]*len(key_word_list)
            for j in range(len(cleaned_tokens_list[i])):
                for k in range(len(key_word_list)):
                    if cleaned_tokens_list[i][j] == key_word_list[k]:
                        key_word_table[k] = 1
            all_key_word_list.append(key_word_table)
        #=======================================
        
        all_pos_words = get_all_words(cleaned_tokens_list)
        freq_dist_pos = FreqDist(all_pos_words)
        frquency_list = freq_dist_pos.most_common(500)
        print(str(frquency_list))
        file.write(str(frquency_list)+'\n')
    file.close()



    upload_to_db_source = []
    upload_to_db_use = []
    
    with  open(file_name,"r",encoding="utf8") as openfileobject:
        counter = 0
        for line in openfileobject:    
            use_dict = {}
            line = json.loads(line)
            vader_result = analyzer.polarity_scores(line["text"])
            line["Sentiment_Analysis"] = vader_result
            line["cleaned_tokens"] = cleaned_tokens_list[counter]
            line["key_word_table"] = all_key_word_list[counter]
            use_dict["date"] = line["date"]
            use_dict["Sentiment_Analysis"] = vader_result["compound"]
            use_dict["key_word_table"] = all_key_word_list[counter]
            upload_to_db_source.append(line)
            upload_to_db_use.append(use_dict)
            counter += 1
    
    print(upload_to_db_use)

    upload_to_db_use_final = []
    date_list = []
    for item_1 in upload_to_db_use:
        item_1["key_word_table"] = np.array(item_1["key_word_table"])
        if item_1["date"] not in date_list:
            date_list.append(item_1["date"])
    for date in date_list:
        modified_dict = {}
        Sentiment_Analysis_sum = 0
        key_word_table_sum = np.zeros(11)
        counter = 0
        for item_2 in upload_to_db_use:
            if item_2["date"] == date:
                Sentiment_Analysis_sum += item_2["Sentiment_Analysis"]
                key_word_table_sum += item_2["key_word_table"]
                counter += 1
        modified_dict["date"] = date
        modified_dict["Sentiment_Analysis"] = Sentiment_Analysis_sum/counter
        modified_dict["key_word_table"] = key_word_table_sum.tolist()
        upload_to_db_use_final.append(modified_dict)
    print(upload_to_db_use_final)
    print("Preprocess Done!")
    
    #upload to mongoDB if no need upload, please comment rest of code
    try:
        client = pymongo.MongoClient("mongodb+srv://foreverznb:ZHUNINGBO1234@cluster0.lbeda.mongodb.net/sample_airbnb?retryWrites=true&w=majority")
    except pymongo.errors.ConfigurationError:
        print("An Invalid URI host error was received. Is your Atlas host name correct in your connection string?")
        sys.exit(1)    
    db = client.myDatabase
    my_collection = db["2020-2021_Tweet"]
    my_collection_1 = db["2020-2021_PreprocessedData"]
    try:
        my_collection.drop()  
        my_collection_1.drop()
    except pymongo.errors.OperationFailure:
        print("An authentication error was received. Are your username and password correct in your connection string?")
        sys.exit(1)      
    try: 
        result = my_collection.insert_many(upload_to_db_source)    
        result_1 = my_collection_1.insert_many(upload_to_db_use_final) 
    except pymongo.errors.OperationFailure:
        print("An authentication error was received. Are you sure your database user is authorized to perform write operations?")
        sys.exit(1)
    else:
        inserted_count = len(result_1.inserted_ids)
        print("I inserted %d documents." %(inserted_count))
      
        print("\n")
    