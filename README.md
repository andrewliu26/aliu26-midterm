# Amazon Movie Reviews Rating Prediction

## Methods and Implementation

### 1. Data Loading and Setup

The project begins by loading and preparing the data, verifying that files are accessible in the `data/` directory. A distribution plot of scores in the training set ensures that the data is ready for preprocessing.

### 2. Feature Engineering

We performed feature engineering on the review data, creating new features to enhance model performance:

- **Helpfulness Ratio**: Calculated as `HelpfulnessNumerator / HelpfulnessDenominator`, with missing values filled as 0.
- **Helpfulness Bucket**: Categorizes helpfulness into 'bad', 'medium', or 'good' based on threshold ratios, making it a categorical feature for better interpretability.
- **Relative Time Feature (`YearsAgo`)**: Converts the timestamp of the review into years since the review was posted.
- **Text and Summary Length**: Measures the word count in `Text` and `Summary` to capture review verbosity.
- **Caps Lock Words Count**: Counts words in all caps (more than one letter), as they often signify emphasis in user sentiment.
- **User Rating Habits**:
  - `UserAvgRating`: Average rating given by each user.
  - `UserRatingVariance`: Variance in user ratings to capture consistency.
  - `UserReviewCount`: Total number of reviews written by the user.

These features were processed through `add_features_to()` and `user_stats` for aggregated user-based statistics.

### 3. TF-IDF Vectorization

We use **TF-IDF** to capture key terms and phrases in reviews, transforming the `Text` field into a matrix of TF-IDF scores. Limited to the top 200 words, TF-IDF enables the model to analyze important textual patterns within the reviews.

### 4. Feature Scaling and Combination

The non-TF-IDF features were scaled using **StandardScaler** to ensure they are on the same scale. These scaled features were then combined with the TF-IDF matrix to form the final feature set (`X_train_final`, `X_test_final`, and `X_submission_final`).

### 5. Class Weight Calculation

Given the imbalance in rating distributions, class weights were computed for each score using `class_weight.compute_class_weight` to adjust the model’s attention toward less frequent classes.

### 6. Custom Balanced Weighted KNN Model

A **Balanced Weighted KNN** classifier was implemented to incorporate both distance weighting and custom class weighting. The class weighting is scaled by an `alpha` parameter, which was tuned to optimize the model’s sensitivity to each rating class.

Key components of this model:
- `neighbors_weights`: Adjusted by both distance and class weight.
- `class_votes`: Predicted by summing weighted votes from each neighbor.

### 7. Cross-Validation and Hyperparameter Tuning

Using **Stratified K-Fold Cross-Validation**, we tuned key parameters:
- `n_neighbors`: Number of neighbors considered (set to 200 after initial testing).
- `alpha`: Controls class weight influence (set to 0.25).

Due to time constraints, we fixed parameters to these values based on initial experimentation.

### 8. Final Model Evaluation

The model was trained on `X_train_final` with the best-tuned parameters and evaluated on `X_test_final`. Results include:
- **Accuracy Score**: Evaluates overall model performance on the test set.
- **Confusion Matrix**: Provides insights into the model’s performance for each class.

### 9. Prediction and Submission

The model predictions on `X_submission_final` were stored in a CSV file following the format required for submission. This file includes two columns:
- `Id`: Unique identifier for each review in the test set.
- `Score`: Predicted rating for each review.

## Results and Findings

- **Final Accuracy on Testing Set**: 61.7% (approx.) on the held-out test set.
- The model achieved the best performance with `n_neighbors=200` and `alpha=0.25`.
- Feature engineering contributed significantly to capturing user and review behaviors, which was reflected in model performance.

## Dependencies

To run this project, install the following packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install these libraries with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
