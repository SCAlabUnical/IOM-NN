# IOM-NN
## Iterative Opinion Mining using Neural Networks
<div style="text-align: justify">
Social media analysis is a fast growing research area aimed at extracting useful information from social media platforms. This paper presents a methodology, called IOM-NN (#Iterative Opinion Mining using Neural Networks#), for discovering the polarization of social media users during election campaigns characterized by the competition of political factions. The methodology uses an automatic incremental procedure based on feed-forward neural networks for analyzing the posts published by social media users. Starting from a limited set of classification rules, created from a small subset of hashtags that are notoriously in favor of specific factions, the methodology iteratively generates new classification rules. Such rules are then used to determine the polarization of people towards a faction. The methodology has been assessed on two case studies that analyze the polarization of a large number of Twitter users during the 2018 Italian general election and 2016 US presidential election. The achieved results are very close to the real ones and more accurate than the average of the opinion polls, revealing the high accuracy and effectiveness of the proposed approach. Moreover, our approach has been compared to the most relevant techniques used in the literature (sentiment analysis with NLP, adaptive sentiment analysis, emoji- and hashtag- based polarization)
by achieving the best accuracy in estimating the polarization of social media users.

## How to cite
Belcastro, L., Cantini, R., Marozzo, F., Talia, D., & Trunfio, P. (2020). Learning political polarization on social media using neural networks. IEEE Access, 8, 47177-47187.


CONTINUE...


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and 
testing purposes.

### Prerequisites

```
- Python 3.7
```

### Installing
- Install requirements
```
pip install requirements.txt 
python -m spacy download en_core_web_lg
```
### Use
- Run the HASHET model
```
python run.py
```

## Dataset

The dataset available in the `input/` folder is a sample of 100 tweets which has the sole purpose of showing 
the functioning of the methodology. Each tweet is a json formatted string.

The real datasets on which HASHET has been validated are in the `used_dataset` folder.
In accordance with Twitter API Terms, only Tweet IDs are provided as part of this datasets. 
To recollect tweets based on the list of Tweet IDs contained in these datasets you will need to use tweet 
'rehydration' programs.

The resulting json line for each tweet after rehydration must have this format:
```
{
   "id":"id",
   "text":"tweet text",
   "date":"date",
   "user":{
      "id":"user_id",
      "name":"",
      "screenName":"",
      "location":"",
      "lang":"en",
      "description":""
   },
   "location":{
      "latitude":0.0,
      "longitude":0.0
   },
   "isRetweet":false,
   "retweets":0,
   "favoutites":0,
   "inReplyToStatusId":-1,
   "inReplyToUserId":-1,
   "hashtags":[
      "hashtag"
   ],
   "lang":"lang",
   "place":{      
   }
}
```

## Parameters
`constants.py` contains all the parameters used in the methodology. Changing them will influence the obtained results.
It is recommended to change `W2V_MINCOUNT` and `MINCOUNT` values for larger datasets.

</div>
