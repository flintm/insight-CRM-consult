

## Introduction
As an Insight Data Science Fellow, I've been consulting with startup ('EdK5') that creates and distributes K-5 educational content. EdK5 has been dealing with an issue that is common to growing subscription-based BLAH: their user base is growing (yay!), their customer support personnel can't keep up with the amount of feedback coming in (eek!) and their small data science/engineering team doesn't have a lot of time to build up solutions (drat). 

I was thrilled to get this opportunity, as EdK5's mission hit close to home: I went to a small elementary school and learned science from an analog predecessor, and have spent the last several years working in the educational sector as a professor of structural engineering at Virginia Tech. My research has spanned scales from micro (transport in porous media) to macro (bridge networks), and I believed that my skills in X, Y, and Z would allow me to (a) solve their problem, and (b) provide additional insights to BLAH. You can find slides telling this story [here](nolink).

(diagram of regex/nlp skills, pedagogy background, decision focus/analytics/elicitation) -> happy client

## Problem formulation
Problem formulation was the trickiest part of this consulting project.

### Client goals

1. Characterize the performance of their existing heuristic (hand-built decision tree) method for pre-classifying feedback topics and the customer feedback data itself.
2. Label feedback as related to certain topics (known and as-yet-discovered), ideally in such a way that it could build into their existing decision tree.
3. Classify tickets as "urgent" (i.e., requiring customer support personnel to look at and respond to the ticket) and non-urgent.

### My initial plan of attack
These tasks and an initial overview of the data available suggested that I augment the existing decision tree and with some additional data from the client's customer relationship management system (see next section) to develop a labeled set for supervised learning of topics and/or urgency. I could then use the labeled set with some unsupervised learning (topic modeling and clustering) to generate features for either (a) an urgency classifer or (b) a multi-classifier using a flat or structure, or (c) multi-classifier using a hierarchical structure.

### Plan D, E, F, ...
It became evident through data exploration that there was very little to work with to use directly as supervised labels or to bootstrap to use transfer learning to infer labels. This meant taking a big step back towards using unsupervised methods and a lot more NLP to try to get something out of the data, and then combine it with heuristics.

## Data acquisition and basic pre-processing
The data came in with reasonable structure (CSVs) which could be merged. That said, no one at my client's firm had looked at the data as a whole, which mean that basic exploration was an important first task.

(figure of data overlaps-Venn diagram?)

Plans A, B, and C went out the window when I was able to take a look at a few comments. The comments below are composites but illustrate the sort of quirks present in the data.


## Feature selection and engineering
### Non-text features
The non-text-based features available included:

* Rating (`int`): Rating of individual lession by a customer based on a prompt given at the end of the lesson. Most ratings were high (4 or 5) and did not include a comment. Client insight suggested that ratings of 4-5 and 1-3 were meaningful groupings.
* Lesson and unit (`categorical`): Overall lessons were rated very high thought some were known to be problematic.
* Grade range (`ordinal`): 
* Average rating (`float`): Average across customers/sessions from CRM as of date data was pulled (can go stale).
* Number of ratings shown (`int`): Don't know what this is/trust it
* Lesson duration (`timedelta`):
* Activity duration (`timedelta`):


### Text metadata / basic NLP
The `comment` (blank textbox) field was expected to be the most BLAH. The metadata I considered was:

* Counts: 
	* characters (`int`): Client feedback suggested that long comments (>40-50 words) were meaningfully different, i.e., more likely to provide interesting or urgent feedback.
	* words (`int`):
	* sentences (`int`):
	* special characters (i.e., '!' or '?') (`int`):
	* parts of speech (i.e., verb, noun) (`int`):
* Keywords from pre-classification rubric or other supervised keywords  (`int`/`bool`):
* Sentiment (`float`): depending on package used, could be polarity, positivity, negativity, neutrality, a compound metric, subjectivity, etc.

### Non-basic NLP (engineered features)
The
 
* Word embeddings
* Keyword embeddings

### Other engineered features
* Difference from average lesson rating (`float`/`bool`): 
* Difference from average teacher rating (`float`/`bool`):
* 


## Modeling: from topic clusters to classifier

### All things clusters
As is common in data science projects, the real story of BLAH over the last three weeks is looping, iterative, branching, and BLAH (i.e., not easily representable in the linear format required by a blog post). In the end, the clustering methods I used did [not help]. That said, I thought others might benefit from learning about how naive implementations of various clustering methods did or did not work for this data set.

| Method  | Why | Test Run  | Scaled Run | Full Run | Notes |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | 
Topic model (**LDA**) on **comment bag of words** | Commonly used topic model; bag of words requires least effort in pre-processing; potential to identify keywords to be used later. | **10k comments** returned ambiguous or overlapping clusters, i.e., they were all "great", "fun",...| **10k? 1/2/3-star comments** had better segmentation but the topics were not interpretable to expected clusters. Playing with settings for max/min vocab was not impactful.| **10k 1/2/3-star comments w/ custom stop** only worked with negative only| Even with more custom NLP pre-processing and to continue to segment on rating. The LDA results motivated me to look at options for semi-supervised methods in which I could seed clusters or use a hierarchical clustering model. |
**DBSCAN clustering** on **comment metadata** + **review features** using **cosine similarity**| faster than hierarchical models and good with noisy data; I have sufficient number of reviews especially if reviews are expanded by sentences; selects its own number of clusters; cosine similarity works for high-dimensional data | **2k rich metadata + features** clustered the reviews into 'short positive/neutral', 'short negative', and 'long' clusters, which was sufficiently promising for me to continue. | **10k metadata + features** even with dropped part-of-speech metadata this worked quite well, producing 6 clusters with some segmentation into positive/negative, video errors, and other lesson problems. still blurry | **100k metadata + features + topics** clusters very small compared to noise. | Some sort of textual data is clearly required--again, either NLP keyword counts, topic flags, or word embeddings. My improved DT did give some coherence with the full set |
**Random Forest unsupervised** | very common; requested by client given belief that decision tree is appropriate; if successful unsupevised can translate to supervised | | |  |

Topic number 0
['prep', 'hands', 'materials', 'kids', 'year', 'wasn', 'experiment', 'bit', 'work', 'didn']


Topic number 1
['way', 'just', 'able', 'long', 'end', 'sure', 'difficult', 'grade', 'videos', 'video']


Topic number 2
['follow', 'long', 'longer', 'hard time', 'steps', 'class', 'took', 'lot', 'hard', 'time']


Topic number 3
['maybe', 'don', 'just', 'different', 'water', 'video', 'good', 'think', 'little', 'kids']


Topic number 4
['trouble', 'glue', 'small', 'wax paper', 'wax', 'snowflakes', 'directions', 'need', 'graders', 'paper']

----
without stop word removal
Topic number 0
['confusing', 'science', 'activity', 'steps', 'difficult', 'better', 'mystery', 'directions', 'video', 'students']


Topic number 1
['wax', 'liked', 'loved', 'read', 'snowflakes', 'videos', 'graders', 'kids', 'activity', 'paper']


Topic number 2
['lot', 'maybe', 'different', 'long', 'class', 'activity', 'students', 'took', 'hard', 'time']


Topic number 3
['grade', 'just', 'water', 'good', 'bit', 'great', 'experiment', 'little', 'students', 'lesson']


Topic number 4
['used', 'use', 'don', 'think', 'really', 'lesson', 'like', 'work', 'kids', 'didn']
---
Topic number 0
['doug', 'game', 'little', 'know', 'love', 'confusing', 'science', 'directions', 'mystery', 'students']


Topic number 1
['wasn', 'good', 'just', 'video', 'videos', 'like', 'great', 'kids', 'activity', 'lesson']


Topic number 2
['student', 'use', 'different', 'maybe', 'students', 'took', 'just', 'way', 'long', 'time']


Topic number 3
['took', 'kids', 'don', 'longer', 'got', 'year', 'materials', 'water', 'lot', 'experiment']


Topic number 4
['directions', 'students', 'idea', 'cars', 'couldn', 'didn work', 'used', 'video', 'work', 'didn']


Topic number 5
['glue', 'wax paper', 'wax', 'hard time', 'snowflakes', 'students', 'time', 'graders', 'paper', 'hard']


Topic number 6
['grade', 'lot', 'steps', 'lesson', 'time', 'class', 'needed', 'activity', 'difficult', 'students']
### Classification


## Validation
Ran original DT and my modified DT (v2) on full population for one topic cluster (prep)

| Metric | MF-DT | DT |
| ------ | ----- | -----|
| Precision | 0.77-0.78 | 0.89 |
| Recall | 0.84-0.94 | 0.73-0.88 |
| f1 | 0.4-0.42 | 0.40-0.44 |
| Total classified | 325* | 397 |
| No. clusters | 4 | 1 |
*Up to 7109 if the additional cluster of 'activity tip' is considered. 

Selection of reviews for validation loosely used a stratified sampling technique, with 243 entries analyzed as of the time of this blog post. The reviews were randomly sampled within the following sets:

* Reviews where both DTs flagged prep volume: all.
* Reviews where only my DT flagged prep volume: all.
* Reviews where only the original DT flagged prep (and none of my prep sub-topics were flagged): I labelled 25% of these.
* Reviews which contained the phrase " prep" but were not flagged by either DT, i.e., possible false negatives: I labelled 3% of these (total of 41).
* Reviews which did not contain " prep" and were not flagged, i.e., possible true negatives: I labelled 0.03% of these (total of 50).

These steps yielded a balanced set of 114 true positives (strictly on prep volume), 16 ambiguous positives (about prep but could be material volume instead of time), and 113 true negatives (not related to prep at all).


## Recommendations

* Build new decision tree / classifier using XYZ
* Reconsider accepting false positives given high volume of false negatives
* Label all reviews seen by CS to build up set for supervised learning
* Periodically check macros book, re-run XYZ to generate updates to new classifier / check performance.


## Conclusions
My consulting project resulted in identifying XYZ. [Limitations]. Going forward, to productionize, ... Stale data, etc.

In addition to XYZ, my client asked for a Jupyter notebook BLAH, and any insight. In addition to a curated notebook BLAH, several insights. 

Generalized. 

## Acknowledgements

Final thanks to my contact at EdK5, who was XYZ.

Kudos to Yuwen XX, whose blog post organization I followed.
