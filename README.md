# 678_AntCluster
+ Ant
    * pharomone == attributes
    * similarity threshold -> same nest
    * memory of met ants
    * threshold = (mean_sim(i, .) + max(sim(i, .))) / 2
    * M_i = size of current nest
    * M_i+ = confidence of belonging to current nest
    * A_i = # of meetings

+ Interaction
    * A, B, no nest, different - ignore
    * A, B, no nest, accepts(mutual) - from nest
    * A no nest, B has nest, A accepts(mutual) B -> nest A = nest B
    * A has nest, B has nest, A accepts(mutual) B, nest A != nest B -> nest small = nest big

+ Variable
    * M_i = size
    * M_i+ = confidence
    * A_i = # of meetings
