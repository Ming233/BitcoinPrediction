import twint
import argparse

def main():
    
    parser = argparse.ArgumentParser(description="Bayesian Refinement Approximation Training Program")
    # parser.add_argument('--path', required=True, help='root of the model')
    
    # model args
    parser.add_argument("--key_words", type=str, default ='bitcoin', help="The key words for search.")
    parser.add_argument("--store_type", type=str, default='json', help="store type, csv or json.")
    parser.add_argument("--min_like", type=int, default=100, help="min number likes for tweet to search.")
    parser.add_argument("--output_file", type=str, default="2020-2021_data.json", help="output file name.")
    parser.add_argument("--time_since", type=str, default='2020-01-01', help="search since time, exp'2019-01-01'")
    parser.add_argument("--time_until", type=str, default='2021-01-01', help="search until time, exp'2020-01-01'")
    args = parser.parse_args()
    print("Training args:", args)
    
    c = twint.Config()
    c.Search = args.key_words
    if args.store_type == 'json':
        c.Store_json = True
    else:
        c.Store_csv = True
    c.Custom = {'tweet': None, 'user': None, 'username': None } 
    c.Format = "User: {username} |Text: {tweet} |Likes: {likes} |RT: {retweets} |Time: {date} {time}"
    c.Min_likes = args.min_like
    #c.Output = "2014-2015_data.csv"
    c.Output = args.output_file
    c.Since = args.time_since            #(string) - Filter Tweets sent since date, works only with twint.run.Search (Example: 2017-12-27).
    c.Until = args.time_until
    twint.run.Search(c)

if __name__ == "__main__":
    main()