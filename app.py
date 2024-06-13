# streamlit run app.py

#import libraries

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #fff;
    color: #000;
    transition: .3s
}

div.stButton > button:hover {
    background-color: #094613 ;
    color: white;
}

</style>""", unsafe_allow_html=True)

# Import dataset

data = pd.read_csv("ALL LAWYERS - Sheet1.csv")


# Pre-process the dataset

data['Location'].replace('chennai','Chennai',inplace=True)
data['Enrollment number'].replace('--','NA',inplace=True)
data['Success rate'] = data['Success rate'].str.rstrip('%').astype(int)

lawyer_data = data.copy()

data['Specialization'].replace({'Arbitration':0,
                                'Civil':1,
                                'Consumer Court':2,
                                'Criminal':3,
                                'Divorce':4,
                                'Family':5}, inplace=True)

data['Location'].replace({'Chennai':0,
                          'Coimbatore':1,
                          'Dindigul':2,
                          'Erode':3,
                          'Trichy':4}, inplace=True)


# feature extraction

features = ['Specialization', 'Experience', 'Location', 'Success rate', 'Avg .Rating']
X = data[features].values

similarity_matrix = cosine_similarity(X, X)


# Create the streamlit app

st.title('Lawyer Recommendation System')
st.write('Please provide the following information for recommendation')


# Collect inputs

user_specialization_category = st.selectbox('Specialization', ('Arbitration','Civil','Consumer Court','Criminal','Divorce','Family'))
specialization_mapping = {
    'Arbitration': 0,
    'Civil': 1,
    'Consumer Court': 2,
    'Criminal': 3,
    'Divorce': 4,
    'Family': 5
}
user_specialization = specialization_mapping.get(user_specialization_category)

user_experience = st.number_input('Experience (in Years)', min_value=0, max_value=120, step=1)

user_location_category = st.selectbox('Location', ('Chennai','Coimbatore','Dindigul','Erode','Trichy'))
location_mapping = {
    'Chennai': 0,
    'Coimbatore': 1,
    'Dindigul': 2,
    'Erode': 3,
    'Trichy': 4
}
user_location = location_mapping.get(user_location_category)

user_success_rate = st.number_input('Success Rate (in %)', min_value=0, max_value=100, step=1)

user_avg_rating = st.number_input('Average Rating (1 to 5)', min_value=1.0, max_value=5.0, step=0.1, format="%.1f", value=4.0)

user_requirements = [user_specialization, user_experience, user_location, user_success_rate, user_avg_rating]


# Calculate similarity and filter

user_similarity = cosine_similarity([user_requirements], X).flatten()

ranked_lawyers = pd.Series(user_similarity, index=data.index).sort_values(ascending=False)

ranked_lawyer_dataset = data.loc[ranked_lawyers.index]

filtered_lawyers = ranked_lawyer_dataset[(ranked_lawyer_dataset['Specialization'] == user_requirements[0]) & (ranked_lawyer_dataset['Location'] == user_requirements[2])]

top_lawyers_data = lawyer_data.loc[filtered_lawyers.index]

sorted_lawyers = top_lawyers_data.sort_values(by=['Experience', 'Success rate', 'Avg .Rating'], ascending=[False, False, False])

sorted_lawyers = sorted_lawyers.reset_index(drop=True)
sorted_lawyers.index = sorted_lawyers.index+1
sorted_lawyers.drop(columns=['S.NO'], inplace=True)

sorted_lawyers['Avg .Rating'] = sorted_lawyers['Avg .Rating'].apply(lambda x: '%.1f' % x)


# Result

submit = st.button('Submit')

if submit:
    st.header("Recommendations for You")
    st.table(sorted_lawyers)
    sorted_lawyers.head()