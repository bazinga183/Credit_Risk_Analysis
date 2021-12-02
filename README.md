# Credit_Risk_Analysis

## Overview of the loan prediction risk analysis:
The purpose of this analysis is to go through a [csv file](https://github.com/bazinga183/Credit_Risk_Analysis/blob/main/LoanStats_2019Q1.csv) that outlines credit risk within certain customers and it is my responsibility to build out a machine learning model to see if it can properly predict credit risk. I am to parse the data using Jupyter Notebook and organize the data using Pandas so that I can properly fit the data into machine learning models.
I did this by converting any data that stored strings into dummy variables so that the model could use these variables as predictors within the model and also dropped the "loan status" column from the DataFrame to to use as my y and kept the remaining columns and converted columns as my X. I split the X and y into training and test sets for the models, and then I used an oversample and undersample algorithm to resample the dataset to eventually generate a classification report. 
The two machine learning models that I used were Random Forest and Easy Ensemble.

## Results:

![imbalanced](https://user-images.githubusercontent.com/46951897/144204515-9579d806-b102-47d8-a911-e00356172358.png)
The Naive Random Oversampling classification report demonstrates a high precision for low-risk and a decent precision and recall for high-risk credit, howeverm the precision for high-risk credit is almost 0 makeing it a poor predictor as it overchooses false positives.

![smote_oversample](https://user-images.githubusercontent.com/46951897/144205487-1c115764-0bd3-49c0-a2cb-5e07b2f3cbe9.png)
For the SMOTE oversampling classification report, it shows similar results but has a slightly better recall for high risk and lower recall for low-risk. In addition, the precision for high and low-risk remained the same, so overall it is still not a great predictor.

![undersample](https://user-images.githubusercontent.com/46951897/144206097-395c860d-014d-4032-9e28-babde4d1d486.png)
For undersampling, we see that the precision remains the same as before for low and high-risk, but recall for both low and high-risk credit also went down.

![over_under_sample](https://user-images.githubusercontent.com/46951897/144371615-fd7d0291-550b-440e-bbfe-521c44f88e26.png)
Here, we observe that the combination of over and undersampling yields the same results as oversampling by itself, the recall is .75 for high risk credit, but the precision is yet again at .01 which makes it unreliable for picking only true positive high-risk data points.

![random_forest](https://user-images.githubusercontent.com/46951897/144371145-e50e507b-2995-4e95-a7de-80186ce7674d.png)
Using the random forest model reveals that the precision is 0.01 greater than the prior models, which is technically a 100% increase in the precision. However, it is still very low and the recall for high risk is slightly less than the recall rates from prior reports, but the recall for low-risk is slightly higher at over 0.8. However, we are looking for better indicators for denying credit, so the model may still be a little unreliable.

![easy_ensemble](https://user-images.githubusercontent.com/46951897/144372000-ceb1197b-427a-4b44-8d3c-74eb808ebc57.png)
Lastly, the easy ensemble also had a 0.02 precision for high-credit, but both recalls for high and low-credit were much more reliable as they both had over 0.7. This seems to make it the most reliable machine learning model.

## Summary:

In essence, the precision score for all of the models was less than or equal to 0.02 for high-risk credit, which means that there was an abundance of false positives being chosen by the models. The rest of the scores were normally just above 0.5, which makes them randomly reliable, but the easy ensemble and random forest models had the highest precision and the highest average recall scores as well. 
The model I would recommend is the easy ensemble because it had its recall scores for high and low-risk credit at above 0.7 which made it the most reliable model compared to the others and the precision for high-risk credit was also tied for the highest score with random forest. Aside from that, it bolstered the best scores, which makes it the recommendation for this data.
