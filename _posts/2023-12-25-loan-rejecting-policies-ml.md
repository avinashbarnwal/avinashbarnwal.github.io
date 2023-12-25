---
title: "Loan Rejection Policies — Machine Learning"
collection: posts
type: "Data Science"
permalink: /posts/2023-12-25-loan_rejection
date: 2023-12-25
---

Interpretability versus performance trade-off given common ML algorithms. Source — http://tinyurl.com/4n5xtszb

When customers apply for the loans, they are generally accessed on more than 10K variables. Sources could range from on-us performance, bureau, cash flow, and demographic features. This leads to a rich profile of the merchants to assess their creditworthiness.

There could be an interaction effect among the variables that might not get captured in linear models. Therefore, tree-based models are generally more suited to capture the non-linear patterns in the profile of the merchants to assess creditworthiness.

Tree-based models do have higher predictive power compared to linear models but they come with limitations of explainability and transparency. We need better explainability and transparency around machine learning models as we need to explain why we are rejecting customers. Adverse Action codes (AAC) are generally used to provide why customers are getting rejected. Providing robust AAC is very important for customer experience. If a worthy customer is getting rejected but she/he gets a loan from a competitor creates a bad brand perception.

Robust AAC are features that are reliable meaning that similar credit characters will also be rejected because of the same features. Another important characteristic would be if features are changed in a favorable direction, the probability of default will reduce more than random features or correlated features.

Following are the official tests to achieve the goals:-

Evaluation:- Fidelity — The ability to identify features that are the principal drivers of the adverse credit decision. There are two ways we can identify the Fidelity of the models.
- Nearest Neighbor — Same prediction for the neighbors based on the given driver features. If not the same, then Fidelity is low.
- Perturbation Test — Driver features identified have higher impact than a. changing other, randomly selected, features in the model. b. changing other features that are closely correlated with the identified driver features.
Evaluation:- Consistency — Whether driver features identified by the same tool( LIME, SHAP, etc.) across different models or by different tools across the same model vary. It is tested as below:-
- Different Open Source Tools and Same Model — Trying different open-source tools on the same model leading to the same driver variables.
- Same Open Source Tool and Different Models— Trying the same open source tool on different models.
  
In the next part, we will conduct these frameworks on the real dataset — https://www.kaggle.com/competitions/home-credit-default-risk. Stay tuned.
