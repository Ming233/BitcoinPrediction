# Bitcoin prediction Twitter data scraping


## Required packages
Before you run this code you need install all requirement package first:

### [Twint](https://github.com/twintproject/twint)
`pip install twint`

### [NLTK](https://www.nltk.org/)

`pip install --user -U nltk`

### [NLTK](https://github.com/cjhutto/vaderSentiment)

`pip install vaderSentiment`

## How to use our code

1. run Twitter_scrap.py --> raw Twitter data json file
2. put test_data.json into NLTK's library root folder
3. run NLP.py --> formalize and analyze the raw Twitter Data and upload the one-hot keywords and sentiment intensity to MongoDB
