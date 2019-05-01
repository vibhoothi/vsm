# Vector Space Model

## What are different types of models in Information Retrival 
There are two types of models
* Boolean Retrival 
* Vector Space Model

## Disadvantages of Boolean Retrival Model 
  * Similarity function is boolean
    * Exact-match only, no partial matches
    * Retrieved documents not ranked
  * All terms are equally important
    * Boolean operator usage has much more influence than a critical word
  * Query language is expressive but complicated

## What is Vector Space Model 
* In Vector Space Model both Documents and queries are vectors each w(i,j) is a weight for term j in document i
* "bag-of-words representation"
* Similarity of a document vector to a query vector = cosine of the angle between them
* Cosine is a normalized dot product
* Documents ranked by decreasing cosine value
* Formula is ![](https://i.imgur.com/wdlDQQd.png)
 * sim(d,q) = 1 when d = q
 * sim(d,q) = 0 when d and q share no terms
