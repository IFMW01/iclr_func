---
layout: distill
title: The Functional Perspective of Neural Networks
description: Common wisdom suggests that neural networks trained on the same dataset reaching the same accuracy and loss can be considered equivalent. However, when considering neural networks as functional representations of their input space, it becomes clear that neural networks all represent distinct functions that enable the model to have predictive capacity. In this blog post, we review functional perspectives which have been used to understand the success of neural network ensembles on more modern architectures concurrently with a dive deep into existing functional similarity metrics that assess the diversity of neural network functions, showing the pitfalls of just accuracy and loss based perspectives when considering neural network function. 
date: 2025-05-07
future: true
htmlwidgets: true
hidden: false
authors:
  - name: Anonymous

# Anonymize when submitting
# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2025-05-07-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Why the Functional Perspective
    subsections:
    - name: How Can Functions Differ
  - name: Traditional Network Analysis
    subsections:
      - name: Test Error Analysis
      - name: Loss Analysis
      - name: Summary of Traditional Analysis
  
  - name: Qualitative Functional Analysis
    subsections:
      - name: Functional Similarity Visualisations
      - name: Prediction Analysis
      - name: Summary of Qualitative Functional Analysis
  - name: Quantative Functional Analysis
    subsections:
      - name: Activation Distance
      - name: Cosine Similairty
      - name: JS divergence 
      - name: Summary of Quantative Functional Analysis
  - name: Impact of Functional Network Analysis
  # - name: Conclusions
  # - name: Citations
  # - name: Footnotes
  # - name: Code Blocks
  # - name: Diagrams
  # - name: Tweets
  # - name: Layouts
  # - name: Other Typography?

  #   - name: Why the functional perspective
  #   subsections: Re-producing Fort el al.,
  #    - name: Test Error Analysis
  #     - name: Loss Analysis
  #     - name: TSNE PLots
  #     - name: Prediction Disimilairy
  # # - name: Updated work
  # #   subsections: Extending Functional Analysis 
  # #   - name: Cosine Similairty 
  # #   - name: JS Divergence
  # #   - name: Activation Distance
  # # - name: Summary and Key Takeaways

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
    
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

<!-- Note: please use the table of contents as defined in the front matter rather than the traditional markdown styling. -->

##  Why the Functional Perspective

The gensis of regarding neural networks as function machines is owed to the original analysis of ensembling overfit neural networks in the 1990's as a way to reduce the residual generalisation error for two-hidden layer networks performing classification tasks, inspired by notions of fault tolerant computing, wherein a noisy rule is formed by combining many local minima using a collective decision strategy making the resulting output less fallible than any single network<d-cite key="hansen1990neural"></d-cite>; showing how shallow multilayer perceptron (MLP) networks employ different functions to enable improved performance. Fort et al,. popularised this understanding for deep neural networks where they explored the properties of ensembled neural networks via prediction comparisons over just traditional loss and accuracy analysis<d-cite key="fort2019deep"></d-cite>; the work echos the understanding of "noisy" function combination to answer important questions regarding the efficacy of ensembled network performance. 

<a name="func_perspective" id="func_perspective">**The Functional Perspective**</a> : We define the Functional Perspective as any research endevour that attempts to exhaustively characterise the divergence and naunce of all network outputs on a layer of interest in a comaprative fashion using a combination of qualitative and/or quantative lines of enquiry. 

Understanding that neural networks form different functions over their input space is a critial idea that has numerous safety implications. In this blog post we reproduce existing functional analysis conducted by Fort et al., on contemporary vision transformers, providing accuracy, loss, prediction disagreemnt and visual analysis of networks trained on the dataset. We then further the work by introducing the results for the best attempts of contemporary literature to capture function of neural networks<d-cite key="klabunde2023similarity"></d-cite>. We provide a commentary on the pitfalls of traditional analysis of accuracy, loss and visual representations of network functions and outline the importance of culminating such analysis with more quantative methods; closing with a perspective on the wider role functional analysis can have on neural network interpretability and safety in the future. 

<!-- This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine.
You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`.
If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph.
Here is an example:

$$
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
$$

Note that MathJax 3 is [a major re-write of MathJax](https://docs.mathjax.org/en/latest/upgrading/whats-new-3.0.html) 
that brought a significant improvement to the loading and rendering speed, which is now 
[on par with KaTeX](http://www.intmath.com/cg5/katex-mathjax-comparison.php). -->

###  <a name="why_func" id="why_func">How Neural Network Functions Differ</a> 

Neural networks that train on the same data can be considered as functional representation of its input space. As a result models that train on the same data can vary considerably on inputs which leads to different overall behaviour. For example in the figure below we can see two hypothetical models that are trained on the the ten class image classifcation task of CIFAR10 <d-cite key="krizhevsky2009learning"></d-cite>. It is evident that these models will both correctly classifiy the input image as a cat - additionally it can be noted that both models have the same loss value of **1.139**. Considering these two metrics (accuracy and loss) alone could lead to the misconception that these models are functionally equivalent given the absolute similairty of their loss and accuracy. However, when considering the output probailities which represent the function of each model it is evident that the functions are different. 

For model one (left image) the 4 highest prediction probabilities other than the predicted class of cat (0.32) are that of the automobile (0.20), airplane (0.12), ship (0.10) and dog (0.10) - as a result from this output perspective it could be argued that the models function puts this example closer to various vehicles over other animals. On the otherhand for model two (right image) the 4 highest prediction probabilities other than the predicted class of cat (0.32) are that of the dog (0.20), deer (0.12), bird (0.10) and frog (0.10) - this models function puts the input image of a cat closer to other species of animals. 
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/pdf/Model_One.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/pdf/Model_Two.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An example of two hypothetical models trained on CIFAR10 that have equivalent prediction agreement, accuracy and loss on the input space while concurrently having nonequivalent functions.
</div>
When considering the functions of the two modles if we were to use them in deployment - despite the accuracy and the loss being equivalent, it would be reasonable to use model two for these types of inputs as it has a function that better captures the distinction between animals and vehicles. This is an important property of  model two as it could suggest that model would be more robust. While this is a contrived exmaple it is not infeasible that such functions could arise in practice which is why model evaluation should be expanded from loss and accuracy to include quantative functional perspectives.


<!-- To ensure that there are no namespace conflicts, you must save your asset to your unique directory
`/assets/img/2025-05-07-[SUBMISSION NAME]` within your submission.

Please avoid using the direct markdown method of embedding images; they may not be properly resized.
Some more complex ways to load images (note the different styles of the shapes/shadows): -->

## Traditional Functional Analysis

Typically, neural network similarity is considered from an accuracy and loss perspective. In our previous section, we have provided contrived examples of when this approach could be flawed. In this section, we look at more qualitative methods of analysing functional relations of neural networks trained under different conditions on CIFAR10<d-cite key="krizhevsky2009learning"></d-cite> as done by Fort et al,.<d-cite key="fort2019deep"></d-cite> for accuracy and loss. To make the experiments results relevant we use the contemporary architecture of the Vision Transformer<d-cite key="dosovitskiy2020image"></d-cite>. All models have the same training hyperparameters with only two variables (initailisation and training data order) altered. 

The three conditions are as follows:

- **Base**: a base model which is trained on a dedicated seed
- **SIDDO**: a model which has the *same initialisation as the base model but is trained on a different data order*.
- **DISDO**: a model which has a *different initialisation but is trained with the same data order as the base model*.  

Through these three conditions, we show how traditional analysis of accuracy and loss analysis may be misleading. 


### Test Error Analysis

To evaluate the neural networks in the three different conditions based on the accuracy, we employ a landscape visualisation tool <d-cite key="li2018visualizing"></d-cite>, as used by <d-cite key="fort2019deep"></d-cite>, to present both 2D and 3D representations of the test error landscapes.

For the 2D and 3D test error plots below, at the minima, where X and Y coordinates are **(0,0)**, it can be observed that the **SIDDO**, Base and **DISDO** models have very similar test errors of 26.870, 27.020 and 27.240, respectively. Additionally, from both perspectives, it is hard to tell if the models are different, given the similarity between their 2D and 3D visualisations. 

Interact with the figures below and try to gain an understanding of the plots to get an intuitive gauge of the error spaces. 

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-05-07-distill-example/test_error_landscapes.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="caption">
    <b>Two dimensional</b> test error plots in 51 random directions for the X and Y axsis<d-cite key="li2018visualizing"></d-cite>.<b>Left</b> represents a model with the same initailsation trained on a different data order (<b>SIDD0</b>) , <b>middle</b> is the base model (<b>Base</b>) and <b>right</b> is the model trained with a different initialisation but same data order as the base model (<b>DIDD0</b>). 
</div>


<div class="l-page">
  <iframe src="{{ 'assets/html/2025-05-07-distill-example/3d_test_error_landscapes.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="caption">
    <b>Three dimensional</b> test error plots in 51 random directions for the X and Y axsis<d-cite key="li2018visualizing"></d-cite>.<b>Left</b> represents a model with the same initailsation trained on a different data order (<b>SIDD0</b>) , <b>middle</b> is the base model (<b>Base</b>) and <b>right</b> is the model trained with a different initialisation but same data order as the base model (<b>DIDD0</b>). 
</div>

Given the similarity of the test error values and visualisations, one could assume that the models have the same function, with some minute differences, given the subtle misalignments on the 3D plots. However, we know that under these different training conditions, these models should differ in their functional representations.
 
### Loss Analysis

When considering the loss landscape visualisation analysis<d-cite key="li2018visualizing"></d-cite> in the 2D and 3D figures below, we are confronted with similar issues. The 2D and 3D test loss plots below, at the minima, where X and Y coordinates are **(0,0)**, for **SIDDO**, Base and **DISDO** are 1.993, 1.948 and 1.932, respectively. Again, these values are not too dissimilar, and the 2D plots, in particular, suggest the same trend for their loss regions. Once again, we invite the reader to play with the 2D and 3D visualisations of the loss landscapes to get an intuitive feel for what the figures are conveying. 

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-05-07-distill-example/test_loss_landscapes.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="caption">
    <b>Two dimensional</b> loss landscape plots in 51 random directions for the X and Y axsis<d-cite key="li2018visualizing"></d-cite>.<b>Left</b> represents a model with the same initailsation trained on a different data order (<b>SIDD0</b>) , <b>middle</b> is the base model (<b>Base</b>) and <b>right</b> is the model trained with a different initialisation but same data order as the base model (<b>DIDD0</b>). 
</div>
<div class="l-page">
  <iframe src="{{ 'assets/html/2025-05-07-distill-example/3d_test_loss_landscapes.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="caption">
    <b>Three dimensional</b> loss landscape plots in 51 random directions for the X and Y axsis<d-cite key="li2018visualizing"></d-cite>.<b>Left</b> represents a model with the same initailsation trained on a different data order (<b>SIDD0</b>) , <b>middle</b> is the base model (<b>Base</b>) and <b>right</b> is the model trained with a different initialisation but same data order as the base model (<b>DIDD0</b>). 
</div>

On this occasion, when we consider the loss landscape 3D visualisation, statements could be made about the similarity of the loss landscapes for the **SIDDO**, Base and **DISDO** models. **SIDDO** and the Base models appear to have similarly structured loss landscapes when compared against the Base and **DISDO** models. While qualitatively, this suggestion appears reasonable, we explain further in the blog post why this illusion of functional similarity does not hold for quantitative measures. 

### Summary of Traditional Analysis

The neural networks trained in these conditions are similar from the test accuracy and loss perspective. Their test accuracy landscape does not deviate massively with perturbation, and despite the loss of landscapes appearing differently on the 3D visualisation, the 2D visualisations resemble one another. Throughout the rest of this blog post, we argue why this perspective alone is not enough to gauge the functional similarity of the models in these conditions and discuss and present alternate avenues for analysis that yield improved insight. 

## Qualitative Functional Analysis

In this section, we employ the analysis conducted by <d-cite key="fort2019deep"></d-cite> using TSNE<d-cite key="van2008visualizing"></d-cite> that shows functional dissimilarity of neural networks. We further this by inclduing other unexplored visualisation methods of PCA, MDS and Spectral Embedding to support the analysis further. We also explore functional similarity divergence that can be captured at a low fidelity by the Prediction Dissimilarity metric used by Fort et al.,. For both visualisation and prediction dissimilarity, we discuss how these avenues for comparing functions may be misleading and incapable of describing the intricacies of functional divergence in details but do provide a high-level trends which are important to recognise.

### Functional Similarity Visualisations

For the functional analysis of neural networks Fort et al., employed a TSNE embedding visualisation to visualise qualitatively how functions diverge. Their findings showed that although neural networks have similar loss and accuracy, they are represented in different functional spaces. The visualisation served as compelling evidence for neural networks forming different noisy functions <d-cite key="hansen1990neural"></d-cite>. In the recreation of this plot we have used TSNE visualisations and other embeddings (PCA, MDS and Spectral Embeddings) to show the same functional divergence. In the figure below, we plot the function during training, which shows that neural networks become increasingly functionally dissimilar over training. The agreement of this general trend across different embedding strategies shows that this finding is robust. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/compare_t-sne_projection_3d.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/compare_pca_projection_3d.png" %}
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/compare_MDS_projection_3d.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/compare_spectral_embedding_projection_3d.png" %}
    </div>
  </div>
<div class="caption">
    Qualitative visualisations of the **SIDDO**, Base and **DISDO** functions over training. <b>Top-Left</b> TSNE<d-cite key="van2008visualizing"></d-cite>, <b>Top-Right</b> Pinciple Component Analysis<d-cite key="pearson1901liii"></d-cite> , <b>Bottom-Left</b> Multiple Dimensional Scalling<d-cite key="mead1992review"></d-cite>  and <b>Bottom-Right</b> Spectral Embedding<d-cite key="10.5555/2980539.2980616"></d-cite> 
</div>

However, it is important to note that while there is an agreement in the general trend of functional divergence, each of the visualisation strategies shows different training paths - the nuances in each method's functional pathways between **SIDDO**, Base, and **DISDO** could lead to different conclusions which may be an artefact of the specific method. As a result, to have the most informative view when using such qualitative methods, it is essential to use a range of embeddings to confirm overall trends without reading too much into the functional illusions that stem from a particular method. Moreover, it is challenging to make statements on the specific functional proximities of different training conditions on functional similarity via functional visualisation as each of the final functional locations is hard to compare qualitatively and can only be commeteted on subjectively. 



### Predcition Disagreement Analysis

Prediction disagreement quantifies how frequently two or more neural networks have the same classification on the same input. It provides a proxy for understanding which inputs neural networks diverge on and allows one to reason how these networks represent different functions. While it can be considered a more quanatative metric, we regard it as a weak functional similaity as it only conisders the argmax of model instead of measuring the function space of predictions. The figure in the section <a href="#why_func">**How Neural Network Functions Differ**</a> illustrates how this metric may provide functional similarity illusions as models can agree on the final prediction but have a divergent prediction space that can indicate apparent modelling properties absent in this analysis. We include it within the qualitative section of this analysis as it is only capturing a general trend which aids the understanding of functional divergence but does not provide a genuinely quantitative means for evaluating the functions of each model and how they are different as it is a weak measure. One could imagine a scenario in which prediction disagreement could lead to false conjecture on functional similairty for example two models disagree on the final classifcation of a input item such as one model predicting a cat and the other predicting a dog but you only 0.01 between each class, functionally these models could be very similliar over thre prediction space output but would be considred disimillair from a predition disagreement perspective which could be incorrect.  

The figure below depicts how the prediction disagreement changes between the base model and the models in the **SIDDO** and **DISDO** conditions during training. The figure provides an intuitive understanding that each model has different functions, which results in prediction discrepancy, which gets stronger through training. However, despite the pitfalls of this analysis of functional analysis it does reaffirm the notion that accuracy and loss provide a myopic perspective of simialirty that must be explored beyond to understand the properties of individual models.

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-05-07-distill-example/SIDDO_predictions.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
  <iframe src="{{ 'assets/html/2025-05-07-distill-example/DISDO_predictions.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="caption">
 Prediction Disimilairty of <b>SIDDO</b> compared to <b>Base</b> (<b>Left</b>) and <b>DISDO</b> compared to <b>Base</b> (<b>Right</b>) during training - a higher prediction disimailirty indicates less agreement on prediction.
</div>

### Summary of Qualitative Functional Analysis



## Quantative Functional Analysis

In this section of the blog post, we extend the work of Fort et al., to show how quantitative metrics can provide improved insights into the functional similarity of neural networks and how often they tell a disjointed story from that presented by more qualitative lines of analysis. The quantitative metrics selected represent a portion of the available documented functional analysis metrics<d-cite key="klabunde2023similarity"></d-cite>. Akin to the previous section, we use the same architecture, datasets and experimental setups to explore the functional similarity. The only modification is that the **SIDDO** and DISD0 conditions are averaged across three models, which is more feasible because no visualisations are required. In the plots, **Model 1** always refers to the **Base** model. The descision to average the results was made to provide more robustness to the overall analysis and resulting conclusions made in this section. 

For consistency, our calculations of the respective metrics in the figures in this section below are done by comparing each model's output function against every other model and then averaging the metrics per epoch and plotting the resulting metrics values change across training.

### Activation Distance

Activation distance <d-cite key="chundawat2023can"></d-cite>, also reported as the norm of prediction difference<d-cite key="klabunde2023similarity"></d-cite>,  represents the *l2* distance of neural network outputs - from this distance, the predictive distance between neural networks can be better understood quantitatively. A **lower activation distance** closer to 0 indicates less functional deviation between model outputs, while a **higher activation distance** suggests functional divergence. It can be calculated by taking the outputs of two models and averaging the *l2* distance of outputs across input batches, **as shown in the Python code below:**

{% highlight python %}

import torch

def activation_dist_fn(base,compare):
    distances = torch.sqrt(torch.sum(torch.square(sf(base) - sf(compare)), axis = 1))
    return distances.mean().item()

{% endhighlight %}

When we compare the activation distance of neural networks trained in different conditions, it can be observed that neural networks, regardless of **SIDDO** or DISD0 conditions, are dissimilar not only to the base model but to one another. If the activation distance over training remains at or close to 0, one could argue that the models have the same function. However, as we see this activation distance deviate over training, it can be understood that the models move in different functional directions during training. 
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/same_init_different_order_act.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/different_init_same_order_act.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Average activation distance of model outputs on the test set of models through training. <b>SIDDO</b> is presented on the <b>left</b> and <b>DISDO</b> is presented on the <b>right</b>.  <b>Model 1</b> represents the <b>Base</b> model. Higher values indicate increased functional divergence and lower values indicate functional similairty. 
</div>

Moreover, from the above figure,, simple factors can impact the functional similarity of outputted networks. In this instance, models trained in the **SIDDO** condition are less functionally similar than models **DISDO** condition; this suggests that for models to be more functionally similar, the data order is more important than initialisation being the same. It could be feasible that this is a byproduct of <a href="#data_primacy">**The Data Primacy Effect**</a>.

<a name="data_primacy" id="data_primacy">**The Data Primacy Effect**</a> : Is a phenomena that acknowledges the importantace of data order for functional similairty, wherein models that take similar gradient updates during training end up in more local minima that have a closer functional representation. 

As a result, it can be understood that even though these models reach similar overall loss and accuracy, the functions they create are fundamentally determined by the data on which they are trained. From the test 3D loss landscapes produced earlier the models with the **SIDDO** condition would be assumed to be more functionally similar as the visualisations suggest similarity, however, it is the case that **DISDO** can have very different loss landscapes but can resemble similar functions when considering activation distance of predictions. 



### Cosine Similairty

Cosine Similarity is a metric to measure the cosine angle between two vectors. As a result, model outputs can be vectorised and compared to distinguish how similar their outputs are. For the cosine similarity metric, values that tend towards 1 suggest a more similar functional representation, while values close to zero suggest orthogonal outputs and values of -1 represent polar outputs. This provides a computationally inexpensive mechanism for calculating the functional similarity between model predictions. **The Python code below shows how it can be implemented:**

{% highlight python %}
import torch
def cosine_sim_fn(model_1, model_2):
    return cs(sf(torch.tensor(model_1)), sf(torch.tensor(model_2))).mean()
{% endhighlight %}


The figure below for both the **SIDDO** and **DISDO** conditions largely reflects that of the results observed for the activation distance plots. At the start of training, the cosine similarity of the models is high, with a sharp drop off in the initial epochs, followed by a steady decline of cosine similarity during the middle of training, which finishes with a slight increase towards the end of training. The overall trend here is that each of the models have different functions as soon as training begins, and they remain different (albeit with varying values) throughout training. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/same_init_different_order_cs.png" class="img-fluid rounded z-depth-1" %}
    </div>
        <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/different_init_same_order_cs.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Average Cosine similairty of model outputs on the test set of models through training. <b>SIDDO</b> is presented on the <b>left</b> and <b>DISDO</b> is presented on the <b>right</b>.  <b>Model 1</b> represents the <b>Base</b> model. Lower values indicate increased functional divergence and higher values indicate functional similairty.  
</div>

Furthermore, there is an agreement between both activation distance and cosine similarity, which states that models within the **DISDO** are more functionally similar than models in the **SIDDO** condition. For **DISDO**, the final consent similarity value is higher than that of **SIDDO**; additionally, for **SIDDO**, the cosine similarity drops lower **(circa 0.75)** than any value for **DISDO**. The agreement across metrics further suggests the  <a href="#data_primacy">**The Data Primacy Effect**</a>.

### JS Divergence
Jenson-Shanon (JS) Divergence represents a weighted average of KL divergence that can be employed to evaluate between non-continuous distributions  <d-cite key="lin1991divergence"></d-cite> and is leveraged to understand the functional divergence between model outputs. Models with **high functional similarity have values that tend towards 0**, and models that are **less functionally similar have relatively higher values**. However, the distinction is less clear than with other metrics. **The code below details how JS Divergence can be implemented in Python:**
{% highlight python %}

import numpy as np
import torch.nn as nn
from numpy.linalg import norm
from scipy.stats import entropy


def JSD(P, Q):
    P = nn.Softmax(dim=1)(P)
    Q = nn.Softmax(dim=1)(Q)
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return (0.5 * (entropy(_P, _M) + entropy(_Q, _M))).mean()
# Code from https://stackoverflow.com/questions/15880133/jensen-shannon-divergence

{% endhighlight %}

The figure below for both the **SIDDO** and **DISDO** conditions largely reflects that of the results observed for both activation distance and cosine similarity plots. At the start of training, the JS divergence of the models is essentially zero, with a sharp increase in the initial epochs, followed by a steady increase in the middle of training and a slight decrease towards the end of training. The noticeable trend is that each of the models has different functions as soon as training begins, and they remain different throughout training; again there is consistency between the functional distance of all the models in the respective conditions, which strengthen the notion that different functions form through training for different models.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/same_init_different_order_js.png" class="img-fluid rounded z-depth-1" %}
    </div>
        <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/different_init_same_order_js.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Average Cosine similairty of model outputs on the test set of models through training. <b>SIDDO</b> is presented on the <b>left</b> and <b>DISDO</b> is presented on the <b>right</b>.  <b>Model 1</b> represents the <b>Base</b> model. Lower values indicate increased functional divergence and higher values indicate functional similairty.  
</div>

The plots conclude that there is a total agreement between all of the respective quantitative functional similarity measures that models within the **DISDO** are more functionally similar than models in the **SIDDO** condition. Models in **SIDDO** always have the most functional divergence from one another compared to models in the **DISDO** condition. As a result, the results strongly suggest the impacts of  <a href="#data_primacy">**The Data Primacy Effect**</a>.

### Summary of Quantative Functional Analysis

A noticeable trend within quantitative functional analysis of models is that they clearly depict the functional diversity of neural networks trained on the same dataset. The metrics provide a more detailed insight into the functional distance between models, which is more grounded than qualitative approaches, which are open to more subjective interpretation. Additionally, it is interesting to note that while these metrics measure different qualities of functional similarity, they largely agree with general trends of functional analysis, which shows that they provide a more robust perspective of neural network functional diversity. Moreover, a more transparent understanding of functional diversity can be obtained when combined with visualisations. A point of interest that has arisen from the quantitative results is that models in the **DISDO** condition are more functionally similar than models within **SIDDO**, which shines a light on the functional variation derived from different data orders and the impact of  <a href="#data_primacy">**The Data Primacy Effect**</a> .  

# Impact of Functional Network Analysis

Qualitative and quantitative functional analysis of the similarity of neural network outputs is something that is gaining popularity in the machine unlearning domain<d-cite key="chundawat2023can"></d-cite> as a way to verify unlearning. Additionally, recent studies have explored how to leverage functional preservation for neural network compression<d-cite key="mason-williams2024neural"></d-cite> and pruning<d-cite key="mason-williams2024what"></d-cite>, with explorations into understanding knowledge transfer through a functional lens<d-cite key="mason-williams2024knowledge"></d-cite>. 

When considering model safety with respect to the <a href="#func_perspective">**Functional Perspective**</a> we argue that models should be analysed and tested independantly given that functional divergence occurs for networks with the same architetcure trained on the same dataset. As a result, there can be more robust stress testing efforts of neural networks which can lead to more precise operational bound identification. 

Moreover, employing the  <a href="#func_perspective">**Functional Perspective**</a> when considering neural networks, combining qualitative and quantitative functional analysis to understand and compare neural networks, will aid endeavours in interpretability and help avoid common misconceptions of the relation between networks, their accuracy and loss as a way to comapre functional simailrity.



<!-- [^1]: Same initailisation Differenct Data Order -->