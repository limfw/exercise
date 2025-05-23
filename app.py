import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Scenario introduction
scenario_description = """
### 📘 Scenario: HealthGear+ Analytics Task

You are a machine learning analyst at **HealthGear+**, a company that produces wearable health-tracking devices for sports and elderly care. The company wants to expand its analytics capabilities by developing two key solutions:

**Solution 1:** Group users into behavior-based segments to deliver personalized coaching programs and health tips.  
**Solution 2:** Predict whether a user will receive a critical health alert in the next 7 days to enable proactive care.

**Available Features:**
- Daily step count, heart rate variability, sleep duration
- User alert interaction behavior (response to abnormal health alerts)
- Device type (FitBand, MedBand, SleepTracker)
- Age, region, and medical risk score
- 14-day device log metrics, prior alert history
- Sensor noise rate and environmental conditions (air quality, temperature)
"""

# Define questions and expected answers
qa_data = {
    "a": {
        "question": "You are tasked with grouping users to provide personalized health tips using wearable device data. Based on the features available in the dataset, select a clustering algorithm to build meaningful user segments. Justify your choice based on how the algorithm handles the structure of this data.",
        "answer": "Use Gaussian Mixture Model or K-means for numerical clustering if features are well-preprocessed. If data is noisy or unevenly distributed, DBSCAN is more robust. For hierarchical relationships, use hierarchical clustering. Each algorithm has strengths depending on data distribution."
    },
    "b": {
        "question": "The dataset contains users from three device types: FitBand, MedBand, and SleepTracker. These devices collect similar but not identical signals. How should your clustering approach account for the differences without separating users by device?",
        "answer": "Include device type as a categorical feature and normalize signal-based features per device before clustering. This allows one model to detect behavior patterns without being dominated by device-specific differences."
    },
    "c": {
        "question": "You are preparing the dataset for clustering. Describe a complete feature engineering pipeline to handle ordinal values, non-normal numerical data, and overlapping value ranges across devices. What could happen if this step is skipped?",
        "answer": "Apply ordinal encoding to ordinal features, normalize or transform skewed numerical data, and standardize overlapping features like heart rate. Without preprocessing, distances in clustering will be biased, leading to meaningless groups."
    },
    "d": {
        "question": "After running your clustering model, you obtain a set of user groups with no known ground truth. Select and explain two appropriate evaluation metrics to assess the clustering quality. Why might one common metric produce misleading results in this case?",
        "answer": "Use Davies-Bouldin Index and Calinski-Harabasz Index. Silhouette score assumes spherical clusters and may mislead when clusters vary in shape or density, especially with high-dimensional or mixed-type data."
    },
    "e": {
        "question": "You observe that one cluster contains only a small percentage of users but shows very distinct behavior. Should this be considered a problem or an opportunity? Justify your reasoning based on the business context of HealthGear+.",
        "answer": "It could be a high-value or at-risk segment. Small clusters are not necessarily a problem if they represent unique patterns. They may indicate niche behavior worth targeting for specialized coaching or intervention."
    },
    "f": {
        "question": "You are now building a model to predict whether a user will receive a critical health alert in the next 7 days. Choose a supervised learning algorithm from the list: SVM, K-NN, Logistic Regression, or Naive Bayes. Justify your selection based on the data properties and business needs.",
        "answer": "Logistic Regression is interpretable and works well for imbalanced classification. Naive Bayes handles categorical data efficiently. SVM performs well with complex boundaries. Choose based on need for interpretability or accuracy."
    },
    "g": {
        "question": "Two models were trained to predict future health alerts. Based on the metrics below, select which model should be deployed. Justify your answer in the context of HealthGear+’s priority for user safety.",
        "answer": "Model Y has higher recall and F1-score, meaning it catches more true positives. In healthcare, missing an alert is riskier than a false alarm, so Model Y should be preferred despite slightly lower precision."
    },
    "h": {
        "question": "The engineering team has added 10 new features from sensor noise filtering. To avoid overfitting, you decide to reduce the number of input features. Suggest a feature selection approach that fits this scenario and explain why it's suitable.",
        "answer": "Use L1 regularization (Lasso) to shrink irrelevant features or use Recursive Feature Elimination (RFE) with a tree-based model. These methods help retain only the features that contribute most to prediction."
    },
    "i": {
        "question": "Your predictive model assigns high feature importance to one environmental variable. Explain why relying on this alone might be misleading. Provide an example from the HealthGear+ context.",
        "answer": "High importance could reflect correlation rather than causation. For example, poor air quality may correlate with alerts but not directly cause them. Other confounding factors could explain the relationship."
    },
    "j": {
        "question": "Your final model includes 30 features and performs well. However, you're asked to simplify the model for deployment. Propose a strategy to reduce model complexity without compromising predictive accuracy.",
        "answer": "Use backward elimination or Lasso regularization to remove low-impact features while monitoring performance metrics. Keep only those features whose removal significantly reduces model accuracy."
    }
}

# Streamlit app
st.title("HealthGear+ ML Strategy Assignment")

st.markdown(scenario_description)

selected_key = st.selectbox("Select a question (a-j):", options=list(qa_data.keys()))
st.markdown(f"**Question ({selected_key}):** {qa_data[selected_key]['question']}")

student_answer = st.text_area("Enter your answer here:")

if student_answer:
    corpus = [student_answer, qa_data[selected_key]["answer"]]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).item()

    st.subheader("Similarity Score")
    st.write(f"**{similarity_score:.2f}**")

    if similarity_score >= 0.80:
        st.success("✅ Strong match – your answer is highly aligned with the expected answer.")
    elif similarity_score >= 0.50:
        st.warning("⚠️ Partial match – consider revising or expanding your answer.")
    else:
        st.error("❌ Weak match – please review the concepts and try again.")
