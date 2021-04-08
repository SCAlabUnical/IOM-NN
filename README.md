# IOM-NN
## Iterative Opinion Mining using Neural Networks
<div style="text-align: justify">
Social media analysis is a fast growing research area aimed at extracting useful information from social media platforms. This paper presents a methodology, called IOM-NN (#Iterative Opinion Mining using Neural Networks#), for discovering the polarization of social media users during election campaigns characterized by the competition of political factions. The methodology uses an automatic incremental procedure based on feed-forward neural networks for analyzing the posts published by social media users. Starting from a limited set of classification rules, created from a small subset of hashtags that are notoriously in favor of specific factions, the methodology iteratively generates new classification rules. Such rules are then used to determine the polarization of people towards a faction. The methodology has been assessed on two case studies that analyze the polarization of a large number of Twitter users during the 2018 Italian general election and 2016 US presidential election. The achieved results are very close to the real ones and more accurate than the average of the opinion polls, revealing the high accuracy and effectiveness of the proposed approach. Moreover, our approach has been compared to the most relevant techniques used in the literature (sentiment analysis with NLP, adaptive sentiment analysis, emoji- and hashtag- based polarization)
by achieving the best accuracy in estimating the polarization of social media users.

## How to cite
Belcastro, L., Cantini, R., Marozzo, F., Talia, D., & Trunfio, P. (2020). Learning political polarization on social media using neural networks. IEEE Access, 8, 47177-47187.

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
```
### Use
- Run IOM-NN
```
python twitter_opinion_miner.py
```

## Dataset

The available dataset in the `input/` folder contains tweet collected from the state of Colorado before the 2016 US presidential elections.
Unzip it into the `input/` folder before running IOM-NN. Each row of the dataset represents a tweet and is a json strings formatted as follows:
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

</div>
