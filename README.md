## Introduction
As an Insight Data Science Fellow, I've been consulting with a startup ('ElementarySTEM') that creates and distributes K-5 STEM educational content used in classrooms. ElementarySTEM has been dealing with an issue that is common to growing subscription-based companies: their user base is growing (yay!), their customer support personnel can't keep up with the amount of feedback coming in (eek!) and their small data science/engineering team doesn't have a lot of time to build up solutions (drat). 

I was thrilled to get the opportunity to help this company, as ElementarySTEM's mission hit close to home: I went to a small elementary school and learned science from an analog predecessor (science in a box), and have spent the last several years working in the educational sector as a professor of structural engineering at Virginia Tech. My research has spanned scales from micro (transport in porous media) to macro (bridge networks), and I believed that my skills in problem definition, data mining, and modeling would allow me to (a) scaffold a solution their problem, and (b) provide additional insights to improve their customer support strategy going forward. You can find slides telling this story [here](nolink).

(diagram of regex/nlp skills, pedagogy background, decision focus/analytics/elicitation) -> happy client

## Problem formulation
Problem formulation was the trickiest part of this consulting project. My client had the following goals:

1. Characterize the performance of their existing heuristic method for pre-classifying feedback topics and the customer feedback data itself<sup>*</sup>.
2. Label feedback as related to certain topics (known and as-yet-discovered).
3. Classify incoming review tickets as "urgent" (i.e., requiring customer support personnel to look at and respond to the ticket) or non-urgent.

<sup>*</sup>The existing tool was a hard-coded, hand-built decision tree, one branch of which fed into a rules-based multi-classifier based on the presence or absence of certain words in the review.

### My initial plans of attack (A, B, and C)
These tasks and an initial overview of the data available suggested that I augment the existing heuristic classifer with data from the client's customer relationship management (CRM) system to develop a labeled set for supervised learning of topics and/or urgency. Additional unsupervised learning (topic modeling and clustering) could generate features for either (A) an urgency classifer, (B) a multi-classifier using a flat structure, or (C) multi-classifier using a hierarchical structure.

### Plans D, E, F, ...
It became evident through data exploration that there was very little reliable information to work with to use directly as supervised labels or to bootstrap to use transfer learning to infer labels. This meant taking a big step back towards using unsupervised methods and a lot more NLP to try to get something out of the data, and then combine it with the existing heuristics to produce either the classifier itself or at least a data set suitable for supervised learning.

### How should success be measured?
In my original client interviews, my contact indicated that the customer support team would accept a high false positive rate (i.e., getting reviews flagged as about a topic and needing of attention even if they weren't actually about that topic), because the CS team would not want to miss out on a potentially problematic review. In ML terms, this suggests that recall (detecting true positives) is more important than precision (rate of true positives amongst predicted positives). The commonly used F<sub>1</sub> score equally weights precision and recall, so a better combined measure would be the F<sub>2</sub> score, which doubles the weight on recall.

I identified the miss rate (false negative rate) as an additional metric of interest, as missed problematic reviews would mean that dissatisfied teachers would not get a response, or that the response would be long in coming, or that feedback would not be integrated into revised versions of the lessons.


## Data acquisition and heuristic classifier recreation
The data came in with reasonable structure (CSVs) which could be merged. That said, no one at my client's firm had looked at the data as a whole, which mean that basic exploration was an important first task.

(figure of data overlaps-Venn diagram?)

Plans A, B, and C went out the window when I was able to take a look at a few comments. The comments below are composites but illustrate the sort of quirks present in the data.

When these comments were passed through the existing classifier, I observed the following:

(plo)

Unfortunately, the existing heuristic classifier was not in alignment with this view, as it was overly restrictive/overfit, and caused a very high false negative rate--i.e., it missed many problematic reviews, and in missing them ensured that the problematic feedback was unlikely to be seen by the CS team.

## Feature selection and engineering
### Non-text features
The non-text-based features available included:

* Rating (`int`): Rating of individual lession by a customer based on a prompt given at the end of the lesson. Most ratings were high (4 or 5) and did not include a comment. Client insight suggested that ratings of 4-5 and 1-3 were meaningful groupings.
* Lesson and unit (`categorical`): The lessons were generally rated very high, though some were known to be problematic.
* Grade range (`ordinal`): content was labeled as being appropriate for students in Grades 0 (K), 1, 2, etc., or K-5.
* Average rating (`float`): Average across customers/sessions from CRM as of date data was pulled. The exact provenance of this score is unknown (does it lump revised and unrevised versions of lessons together?), and there is some concern about the data going stale.
* Number of ratings shown (`int`): Theoretically, the number of ratings shown on the website for the lesson, again, provenance and propensity to go stale were considered issues.
* Video duration (`timedelta`): length of the videos associated with the lesson.
* Activity duration (`timedelta`): expected time of the activity or activities associated with the lesson.


### Text metadata / basic NLP
The `comment` (blank textbox) field was expected to be the most important in identifying topics and urgency. The text metadata I considered as possible features were:

* Counts: 
	* characters (`int`): Client feedback suggested that long comments (>40-50 words) were meaningfully different, i.e., more likely to provide interesting or urgent feedback.
	* words (`int`): A possible alternative to character counts, could be more stable across the writing styles of different reviewers.
	* sentences (`int`): A more intuitive descriptor for the amount of content present in the comment, but potentially noisy due to differences in punctuation across reviews.
	* special punctuation characters (i.e., '!' or '?') (`int`): Linked to sentiment but faster to process.
	* parts of speech (i.e., verb, noun) (`int`): Possibly characteristic of sentiment or content, e.g., perhaps a "suggestion" might have more nouns than a "kudos" review.
* Keywords from pre-classification rubric or other pre-determined keywords  (`int`/`bool`): Either counts or presence/absence of a keyword. Keywords could be inferred either from the existing heuristic, from topic modeling, from data exploration, or a combination of these strategies.
* Sentiment (`float`): depending on package used, could be polarity/subjectivity or positivity/negativity/neutrality/compound.

### Non-basic NLP (engineered features)
I considered using more advanced NLP strategies, such as word embeddings, but ultimately decided that this approach was impractical for implementation, because the pipeline complexity would be expected to grow significantly and the model would become more likely to become stale. However, data exploration suggested that many reviews would start and end with a positive sentiment but include some criticism in-between. I intuited that some measure of sentiment variability across sentences might capture this dynamic and could be used to identify urgent reviews.

* Sentiment mean across sentences (`float`):
* Sentiment standard deviation across sentences (`float`):
* Sentiment minimum across sentences (`float`):

### Other engineered features
* Difference from average lesson rating (`float`/`bool`): as all lessons had very high average ratings, the difference between a review and the average might indicate something about the urgency of that review.
* Difference from average teacher rating (`float`/`bool`): the 1-5 star ratings are subjective, and rating value might become more meaningful if put into the context of the teacher's normal rating.


## Modeling: from topic clusters to classifier
### Bag of words NLP
The word counts from my initial clustering and classification model runs were obtained using the following vectorization.

```
cv = CountVectorizer(max_df=0.85,
						 stop_words='english',
						 min_df=0.01,
						 ngram_range=(1,2),
                     max_features=200)
                    
```


### All things clusters
As is common in data science projects, the real story of BLAH over the last three weeks is looping, iterative, branching, and BLAH (i.e., not easily representable in the linear format required by a blog post). In the end, the clustering methods I used did [not help]. That said, I thought others might benefit from learning about how naive implementations of various clustering methods did or did not work for this data set.

| Method  | Why | Test Run  | Scaled Run | Full Run | Notes |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | 
Topic model (**LDA**) on **comment bag of words** | Commonly used topic model; bag of words requires least effort in pre-processing; potential to identify keywords to be used later. | **10k comments** returned ambiguous or overlapping clusters, i.e., they were all "great", "fun",...| **10k 1/2/3-star comments** had better segmentation but the topics were not interpretable to expected clusters. Playing with settings for max/min vocab was not impactful.| **10k 1/2/3-star comments w/ custom stop words** did return some clear topics, but did not provide much insight beyond what could be discovered in basic manual data exploration. Topic clarity was highly dependent on the number of topics selected. | Even with more custom NLP pre-processing and segmentation by rating, the LDA results were lackluster, motivating me to look at options for semi-supervised methods in which I could seed clusters or use a hierarchical clustering model. |
**DBSCAN clustering** on **comment metadata** + **review features** using **cosine similarity**| faster than hierarchical models and good with noisy data; I have sufficient number of reviews especially if reviews are expanded by sentences; selects its own number of clusters; cosine similarity works for high-dimensional data | **2k rich metadata + features** clustered the reviews into 'short positive/neutral', 'short negative', and 'long' clusters, which was sufficiently promising for me to continue. | **10k metadata + features** even with dropped part-of-speech metadata this worked quite well, producing 6 clusters with some segmentation into positive/negative, video errors, and other lesson problems. still blurry | **100k metadata + features + topics** clusters very small compared to noise. | Some sort of textual data is clearly required--again, either NLP keyword counts, topic flags, or word embeddings. My improved DT did give some coherence with the full set |


### Classification
``` 
cv = CountVectorizer(max_df=0.87,stop_words=None,
                     #'english',
                     min_df=0.0001,ngram_range=(1,1),
                    max_features=130)
                    (has prep in it
 ```

## Validation
Ran original DT and my modified DT (v2) on full population for one topic cluster (prep)

| Metric | MF-DT | DT |
| ------ | ----- | -----|
| Precision | 0.8 | 0.9 |
| Recall | 0.6-0.8 | 0.5-0.7 |
| f2 | 0.7-0.8  | 0.5-0.8|
| Miss rate | 0.15-0.3 | 0.25-0.4 |
| Total classified | 325* | 397 |
| No. clusters | 4 | 1 |
*Up to 7109 if the additional cluster of 'activity tip' is considered. 

Selection of reviews for validation loosely used a stratified sampling technique, with 520 entries analyzed as of the time of this blog post. The reviews were randomly sampled within the following sets:

* Reviews where both DTs flagged prep volume: all (43; 42 positive).
* Reviews where only my DT flagged prep volume: all (35; 9 were positive, 94 were adjacent).
* Reviews where only the original DT flagged prep (and none of my prep sub-topics were flagged): I labelled 40% of these (118 of 295; 100 positive).
* Reviews which contained the phrase " prep" but were not flagged by either DT, i.e., possible false negatives: I labelled 8% of these (109 of 1373; 19 were positive).
* Reviews which did not contain " prep" and were not flagged, i.e., possible true negatives: I labelled 0.07% of these (117 of 166853; none were positive).

These steps yielded a fairly balanced set of 193 true positives (strictly on prep volume), 52 ambiguous positives (about prep but could be material volume instead of time; these were treated as negatives in the ML classification), and 276 true negatives (not related to prep at all).


## Recommendations

* Build new decision tree / classifier using XYZ
* Reconsider accepting false positives given high volume of false negatives and existing backlog of reviews.
* Label all reviews seen by CS to build up set for supervised learning.
* Periodically check log of macros created by CS team, re-run XYZ to generate updates to new classifier / check performance.


## Conclusions
My consulting project resulted in identifying XYZ. [Limitations]. Going forward, to productionize, ... Stale data, etc.

In addition to XYZ, my client asked for a Jupyter notebook BLAH, and any insight. In addition to a curated notebook BLAH, several insights. 

Generalized. 

## Acknowledgements

Final thanks to my contact at EdK5, who was XYZ.

Kudos to Yuwen XX, whose blog post organization I followed.
