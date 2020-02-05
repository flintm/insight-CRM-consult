# Course-Correct: helping an EdTech startup prioritize customer support actions

*Madeleine Flint; Insight Data Science Fellow, Silicon Valley 2020A;
2020-02-04*

## 1. Introduction
During my time as an Insight Data Science Fellow, I consulted with a startup (which I will call 'ElementarySTEM') that creates and distributes K-5 STEM educational content used in classrooms. I was thrilled to get the opportunity to help this company, as ElementarySTEM's mission hit close to home: I went to a small elementary school and learned science from an analog predecessor of their service (science in a box), and have spent the last several years working in the educational sector as a professor of structural engineering.

ElementarySTEM has been dealing with an issue that is common to growing subscription-based companies: 

* their content is growing (huzzah) their user base is growing (yay!), 
* the set of content reviews is growing exponentially (eek), 
* their customer support team / engineering teams are not growing (as fast), and 
* their customer support personnel can't keep up with the huge volume of valuable feedback (eek). 

My research has spanned scales from micro (transport in porous media) to macro (bridge networks), and I believed that my skills in problem definition and modeling would allow me to quickly scaffold a solution while gleaning useful insights along the way. You can find slides telling this story [here](http://bit.ly/Course_Correct).

(diagram of regex/nlp skills, pedagogy background, decision focus/analytics/elicitation) -> happy client

### 1.1 Problem formulation
Problem formulation was the trickiest part of this consulting project. My client had the following goals:

1. Characterize 3+ years of customer feedback and 1 year customer support data.
2. Assess the performance of their existing heuristic method for labeling reviews as related to a set of topics.<sup>*</sup>
2. Improve, expand, or replace the existing classifier to label reviews as related to new and as-yet-discovered topics.
3. Classify incoming review tickets as urgent or non-urgent, where "urgent" = high priority for customer support to look at and respond to the ticket.

<sup>*</sup>The existing tool was a hard-coded, hand-built decision tree, one branch of which fed into a keyword and rules-based multi-classifier.

#### My initial plans of attack (A, B, and C)
These tasks and an initial overview of the data available suggested that I augment the existing heuristic classifer with data from the client's customer relationship management (CRM) system to develop a labeled set for supervised learning of topics and/or urgency. Additional unsupervised learning (topic modeling and clustering) could generate features for either (A) an urgency classifer, (B) a multi-classifier using a flat structure, or (C) multi-classifier using a hierarchical structure.

#### Plans D, E, F, ...
It became evident through data exploration that there was very little reliable information to work with to use directly as supervised labels or to bootstrap to use transfer learning to infer labels. This meant taking a big step back towards using unsupervised methods and a lot more NLP to try to get something out of the data, and then combine it with the existing heuristics to produce either the classifier itself or at least a data set suitable for supervised learning.

### 1.2 How should success be measured?
In my original client interviews, my contact indicated that the customer support team would accept a high false positive rate (i.e., getting reviews flagged as about a topic and needing of attention even if they weren't actually about that topic), because the CS team would not want to miss out on a potentially problematic review. In ML terms, this suggests that recall (detecting true positives) is more important than precision (rate of true positives amongst predicted positives). The commonly used F<sub>1</sub> score equally weights precision and recall, so a better combined measure would be the F<sub>2</sub> score, which doubles the weight on recall.

I identified the miss rate (false negative rate) as an additional metric of interest, as missed problematic reviews would mean that dissatisfied teachers would not get a response, or that the response would be long in coming, or that feedback would not be integrated into revised versions of the lessons.


## 2. Data wrangling

### 2.1 Acquisition and preprocessing
The data came in with reasonable structure (CSVs) which could be merged. That said, no one at my client's firm had looked at the data as a whole, which mean that basic exploration was an important first task.

![](https://github.com/flintm/insight-CRM-consult/raw/master/images/venn.png "Original vs DT topics")

Plans A, B, and C went out the window when I was able to take a look at a few comments. The comments below are composites but illustrate the sort of quirks present in the data.

When these comments were passed through the existing classifier, I observed that the existing heuristic classifier was not in alignment with this view, as it was overly restrictive/overfit, and caused a very high false negative rate--i.e., it missed many problematic reviews, and in missing them ensured that the problematic feedback was unlikely to be seen by the CS team.

### 2.2 Reproducing the existing classifier

### 2.3 Feature selection and engineering
#### Non-text features
The non-text-based features included standarized metadata about the individual review, target audience, and lesson:

* **Rating** (`int`): Rating of individual lession by a customer based on a prompt given at the end of the lesson. Most ratings were high (4 or 5) and did not include a comment. Client insight suggested that ratings of 4-5 and 1-3 were meaningful groupings.
* **Lesson, unit, and grade range** (`categorical`/`ordinal`): The lessons were generally rated very high, though some were known by the support team to be problematic.
<!--* **Grade range** (`ordinal`): content was labeled as being appropriate for students in Grades 0 (K), 1, 2, etc., or K-5.-->
* **Average rating** (`float`): Average across customers/sessions as stored in the CRM. The exact provenance of this score is unknown (does it lump revised and unrevised versions of lessons together?), and there is some concern about the data going stale.
<!--* **Number of ratings shown** (`int`): Theoretically, the number of ratings shown on the website for the lesson, again, provenance and propensity to go stale were considered issues.
-->
* **Duration** (`timedelta`): expected length of the lesson, multimedia, and/or activities associated with a particular lesson.


#### Text metadata / basic NLP
The `comment` (free text) field was expected to be the most important in identifying topics and urgency. The text metadata I considered as possible features were:

* **Counts**: 
	* **characters** (`int`): Client feedback suggested that long comments (>40-50 words) were meaningfully different, i.e., more likely to provide interesting or urgent feedback.
	* **words** (`int`): A possible alternative to character counts, could be more stable across the writing styles of different reviewers.
	* **sentences** (`int`): A more intuitive descriptor for the amount of content present in the comment, but potentially noisy due to differences in punctuation across reviews.
	* **special punctuation** (i.e., '!' or '?') (`int`): Linked to sentiment but faster to process.
	* **parts of speech** (i.e., verb, noun) (`int`): Possibly characteristic of sentiment or content, e.g., perhaps a "suggestion" might have more nouns than a "kudos" review.
* **Topic keywords** (`int`/`bool`): Either counts or presence/absence of a keyword. Keywords could be inferred either from the existing heuristic, from topic modeling, from data exploration, or a combination of these strategies. See Section 
* **Sentiment** (`float`): depending on package used, could be polarity/subjectivity or positivity/negativity/neutrality/compound.

#### Engineered features
I intuited that variance from the norm or within comments might be a strong predictor of problematic feedback given that reviews and ratings were generally very favorable. Potential engineered features I considered were:

* **Difference from average lesson rating** (`float`/`bool`): as all lessons had very high average ratings, the difference between a review and the average might indicate something about the urgency of that review. I implemented a simple difference; a more sophisticated metric might be the number of standard deviations from the mean.
* **Difference from average teacher rating** (`float`/`bool`): the 1-5 star ratings are subjective, and rating value might become more meaningful if put into the context of the teacher's normal rating.
* **Sentence-level sentiment** (`float`): a measure of the standard deviation of sentiment might see through the "criticism sandwich" effect, and the mean would equalize across a more rational unit of thought (i.e., the sentence as opposed to the word).

I also considered using more advanced NLP strategies, such as word embeddings, but ultimately decided that this approach was impractical for implementation, because the pipeline complexity would be expected to grow significantly and the model would become more likely to become stale. However, data exploration suggested that many reviews would start and/or end with a positive sentiment but include some criticism in-between. 

## 3. Modeling: from topic clusters to classifier
### 3.1 Bag of words NLP
The word counts from my initial clustering and classification model runs were obtained using the following vectorization.

```
cv = CountVectorizer(	max_df=0.85,
						stop_words='english',
						min_df=0.01,
						ngram_range=(1,2),
						max_features=200	)               
```


### 3.2 All things clusters
As is common in data science projects, the real story of my iterative modeling over the last three weeks is both looping and branching (i.e., not easily representable in the linear format required by a blog post). In the end, the clustering methods I used did [not help]. That said, I thought others might benefit from learning about how naive implementations of various clustering methods did or did not work for this data set.

| Method  | Why | Test Run  | Scaled Run | Full Run | Notes |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | 
 **LDA** topic model on **comment bag of words** | Commonly used topic model; bag of words requires least effort in pre-processing; potential to identify keywords to be used later. | **10k comments** returned ambiguous or overlapping clusters, i.e., they were all "great", "fun",...| **10k 1/2/3-star comments** had better segmentation but the topics were not interpretable to expected clusters. Playing with settings for max/min vocab was not impactful.| **10k 1/2/3-star comments w/ custom stop words** did return some clear topics, but did not provide much insight beyond what could be discovered in basic manual data exploration. Topic clarity was highly dependent on the number of topics selected. | Even with more custom NLP pre-processing and segmentation by rating, the LDA results were lackluster, motivating me to look at options for semi-supervised methods in which I could seed clusters or use a hierarchical clustering model. |
**DBSCAN** clustering on **comment metadata** + **review features** using cosine similarity | faster than hierarchical models and good with noisy data; I have sufficient number of reviews especially if reviews are expanded by sentences; selects its own number of clusters; cosine similarity works for high-dimensional data | **2k rich metadata + features** clustered the reviews into 'short positive/neutral', 'short negative', and 'long' clusters, which was sufficiently promising for me to continue. | **10k metadata + features** even with dropped part-of-speech metadata this worked quite well, producing 6 clusters with some segmentation into positive/negative, video errors, and other lesson problems. still blurry | **100k metadata + features + topics** clusters very small compared to noise. | Some sort of textual data is clearly required--again, either NLP keyword counts, topic flags, or word embeddings. My improved DT did give some coherence with the full set |

### 3.3 Translating clustered topics to a multi-topic classifier
In the end, the topic modeling and clustering had only an indirect route to the development of the improved DT and the single-topic classifiers. To generalize, I used the results to:

* confirm the suggestions of my client regarding main categories of feedback
* something about ground truth
* create a schematic/matrix for types of feedback
* select a set of core topics for improved DT development

### 3.4 Single-topic classification using machine learning
In this case I did a bit of hacking to obtain a bag of words that included "prep", as follows.

```
cv = CountVectorizer(	max_df=0.87,  
						stop_words=None,  
						min_df=0.0001,  
						ngram_range=(1,1),  
						max_features=130   )                
```



## 4. Validation
### 4.1 Multi-topic classifier
As shown in the chart below, the improved classifier (bottom) tags a significantly higher number of reviews, while providing more fine-grained detail of the review content. I analyzed the reviews with tagged topics and found that the intersection of topics tagged by both DTs was much smaller than the union. This comparison is made more difficult by my expansion of the topic set from the original 9 to 16, which better reflected what I discovered during the unsupervised data modeling. 

![](https://github.com/flintm/insight-CRM-consult/raw/master/images/Topic_counts.png)

#### Overall performance
To validate the overall performance against some form of ground truth, I randomly selected 200 reviews with comments and tagged them according to an expanded framework. Of these, a little under 20% were related to some topic that was not "kudos", i.e., 2x the number identified by the improved DT but 15x the number identified by the original DT. I interpret this result to mean that there is more room for improvement but that my DT is at least identifying the correct order of magnitude.

#### Topic-of-interest performance (preparation for ML)
I ran my recreation of the original DT as well as my modified DT on the full review set and analyzed data in the "too much prep" cluster.

| Metric | MF-DT | DT |
| ------ | ----- | -----|
| Precision | 0.8 | 0.9 |
| Recall | 0.6-0.8 | 0.5-0.7 |
| F<sub>2</sub> | 0.7-0.8  | 0.5-0.8|
| Miss rate | 0.15-0.3 | 0.25-0.4 |
| Total classified | 325* | 397 |
| Number of clusters | 4* | 1 |
*Up to 7109 if the 5th cluster of 'activity tip' is considered. 

Selection of reviews for validation loosely used a stratified sampling technique, with 520 entries analyzed as of the time of this blog post. The reviews were sampled within the sets shown below, which would be expected to map onto the standard confusion matrix format.

![](https://github.com/flintm/insight-CRM-consult/raw/master/images/ValidationDesign.jpg)

These steps yielded a fairly balanced set of 193 true positives (strictly on prep volume), 52 ambiguous positives (about prep but could be material volume instead of time; these were treated as negatives in the ML classification), and 276 true negatives (not related to prep at all or stating that the prep was easy/short).

### 4.2 Single-topic ML classifier
![](https://github.com/flintm/insight-CRM-consult/raw/master/images/algo_wCat.png)


## 5. Conclusions
### 5.1 Recommendations to client

* Consider employing improved decision tree to aid CS team: classifies many more reviews and is quick to implement (regex).
* Spend some time saved by labeling data within the CRM to serve as ground truth for subsequent development of supervised single-topic classifiers.
* Periodically check log of macros created by CS team, and re-run provided Jupyter notebook to generate updates to new classifier / check performance.
* Reconsider entire strategy/pipeline for identifying urgent reviews and funneling feedback to the content owners/lesson creators.

### 5.2 Recommendations to my future self (or you!)

* The NLP/unsupervised rabbit hole will eat up all your time. Just label a few hundred points to start.
* Learning a version control method for models/Jupyter might have been a worthwhile investment.

### 5.3 Next steps and impact
My consulting project resulted in identifying an existing and potentially critical flaw in my client's current approach to handling free-text customer feedback, and proposed a quick and easy interim solution. Exploration of long term strategies suggests that pretty much any binary classifier can be expected to work well, allowing my client's data science/engineering team to choose an option that they feel has the best tradeoffs between deployment, stability, and latency. Should one of my models or engineered features be of interest, I would caution the team to check for stale data related to averaged ratings and the creep of feedback as new content is added and lessons are revised. 

<!--More practically, I'm hopeful that In addition to XYZ, my client asked for a Jupyter notebook BLAH, and any insight. In addition to a curated notebook BLAH, several insights. -->

More generally, this consulting experience illustrates the advantages of planning for success.

## Acknowledgements
Many thanks to my contact at ElementarySTEM, who was XYZ. I would also like to thank the many great folks who put their project and code snippets online. I pulled code snippets and techniques liberally from the following sources:

* Yuwen
* ...

And, as always, thank you to the generous folks at Insight (mentors, directors, and fellows) for their highly effective suggestions and encouragement along the way.