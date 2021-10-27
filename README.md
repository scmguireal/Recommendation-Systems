# Collaborative Filtering

### Metrics:
* **Adjusted Cos Similarity** (User - user based)
  - Considers different baselines for users
    - ie Some people have different baselines for what makes a 3 star movie and some people will give everything a 5 star rating unless they just really don't like it
  - Measures similarity in terms of their rating for that movie against their average rating. This is looking at the variance from the mean of each users ratings.
    - Sparcity can be a issue here and should be considered
* **Pearson similarity** (User - Item based)
  - Similar to Adj Cos Similarity, but checks the user's rating to all users' ratings for that item
* **Spearman Rank Correlation** (pearson similarity based on ranks not ratings)
  - Instead of using an average rating value for a movie, we'd use its rank amongst all movies based on their average ratings. And, instead of individual ratings for a movie, we'd rank that movie amongst all that individuals ratings.
  - (Can handle ordinal data but that's not practical)
*  **Mean Squared Difference
  - Sums up the square difference of items users x and y rated then divids by number of items in common
  - This is the difference between the two users
    - Similarity between users would be 1/(MSD + 1) (plus 1 to avoid dividing by 0)
* **Jaccard Similarity**
  - Intersection of A and B divided by Union of A and B
  - Really quick way to measure expressions of interest between A and B


**Bleeding Edge**
* https://sites.google.com/view/ruining-he/
* translation-based recommendations
  - Idea is that users are modeled as vectors moving from one item to the next and focuses on hit rate over accuracy
    - Transition space is useful for predicting sequence of events

# Matrix Factorization Methods

* PCA
  - Produces lower dimensional latent features of each row (either (R or R^T) to users(U) or movies(M))
  - R = UEM^T where E is just a diagonal matrix
    - Taking the dot product for associated row in U for the user and columnn in M^T will give you the predicted rating in R if it is not available
* SVD
  - Is just running PCA on both the users and items and giving back the matricies we need in one shot.

* Stocastic Gradient Descent or Alternating Least Squares
  - When values are missing in initial matrix R we can treat that dot product value as a minimization problem that minimizes the error for the known values.
  - Algorithms: SVD++ and Restricted Boltzman Machines

**Bleeding Edge**
* SLIM - Sparse Linear Methods (beat just about everything in recommendations in many different domains)
* The idea behind SLIM is to generate recommendation scores for a given user and item by a sparse aggregation of the other things that user has rated, multiplied by a sparse aggregation of weights, which is where the magic lives.

# Deep Learning Methods

* Restricted Boltzmann Machines
  - https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf
  - Gibbs sampling and contrastive divergence
* AutoRec
  - Autoencoders Meet Collaborative Filtering
  - https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf
* GRU4Rec
  - Session based recommendations with recommendations

**Bleeding Edge**
* GANs for Recommendations
  - RecGAN: Recurrent GANs for Recommendation Systems
  - https://www.brianlim.net/wordpress/wp-content/uploads/2018/08/recsys2018-recgan-recommender.pdf
    - Great for feeding in previous data to generator then giving real data to discriminator for recommending next value in the sequence

# Tensorflow Recommenders (TFRS)

* Retrival Stage
  - Selects recommendation candidates
    - Two Towers Model:
      - Query model
      - Candidate model
    - Multiply the two together to create an affinity
* Ranking Stage
  - Selects best candidates and ranks them

* Multi-task Recommenders
  - Combined (joint) models may perform better than multiple task specific models
  - This means you will have multiple objectives and loss functions

* Deep & Cross Networks
  - Combined features give more insight (ie banana + cookbook --> blender rec)
  - Cross Networks explicitly apply feature crossing at each layer
    - tfrs.layers.dcn.Cross() calls a DCN in tf

**Bleeding Edge**
* Deep Factorization Machines
  - DeepFM: A Factorization-Machine based Neural Netowrk for CTR Prediction
  - Essientially just a hybrid approach combining FMs and deep Networks

       
