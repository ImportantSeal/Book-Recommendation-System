import streamlit as st
st.set_page_config(page_title="Book Recommendation System", layout="wide")

import pandas as pd
import numpy as np
import pickle
import os

# Välimuistitetaan pickle tiedostojen lataus
@st.cache_data(show_spinner=True)
def load_pickle(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        st.error(f"File not found: {file_path}")
        return None

# ladataan pickle-tiedostot
popular = load_pickle('popular.pkl')
pt = load_pickle('pt.pkl')
master_table = load_pickle('master_table.pkl')
similarity_scores = load_pickle('similarity_scores.pkl')

# Jos joku tiedosto puuttuu, lopetetaan sovellus
if popular is None or pt is None or master_table is None or similarity_scores is None:
    st.stop()

# collaborative filtering suositusfunktio, jossa on lisätty filtterit
def recommend_cf_filtered(book_name, min_rating=0.0, min_total=0):
    if book_name not in pt.index:
        return None
    # Haetaan haetun kirjan indeksi pivot taulukosta
    index = np.where(pt.index == book_name)[0][0]
    # lasketaan cosine-similariteetit ja valitaan enintään 10 lähintä naapuria (miinus lukien itse kirja)
    similar_items = sorted(
        list(enumerate(similarity_scores[index])), 
        key=lambda x: x[1], 
        reverse=True
    )[1:10]
    
    recommendations = []
    # Käydään läpi ehdokkaat ja sovelletaan suodattimia
    for i in similar_items:
        temp_df = master_table[master_table['Book-Title'] == pt.index[i[0]]].drop_duplicates('Book-Title')
        if temp_df.empty:
            continue
        avg_rating = temp_df['Average_Rating'].values[0]
        total_ratings = temp_df['Total_number_of_Ratings'].values[0]
        if avg_rating < min_rating or total_ratings < min_total:
            continue
        rec = {
            'Book-Title': temp_df['Book-Title'].values[0],
            'Book-Author': temp_df['Book-Author'].values[0],
            'Image-URL-M': temp_df['Image-URL-M'].values[0],
            'Total_number_of_Ratings': total_ratings,
            'Average_Rating': avg_rating
        }
        recommendations.append(rec)
        if len(recommendations) >= 5:
            break
    return recommendations

# streamlit
st.title("StoryFinder")

# sivupalkissa käyttäjä valitsee suosittelutavan 
option = st.sidebar.selectbox("Select Recommendation Type:", 
                              ("Popularity Based", "Collaborative Filtering"))

if option == "Popularity Based":
    st.header("Popular Books")
    st.write("These books have received the highest ratings and a large number of reviews.")
    
    # suosituimmat kirjat näkyy ruudukossa
    num_books = popular.shape[0]
    for i in range(0, num_books, 5):
        cols = st.columns(5)
        for j, col in enumerate(cols):
            if i + j < num_books:
                row = popular.iloc[i + j]
                with col:
                    st.image(row['Image-URL-M'], width=120)
                    st.markdown(f"**{row['Book-Title']}**")
                    st.write(f"by {row['Book-Author']}")
                    st.write(f"Avg Rating: {row['Average_Rating']}")
                    st.write(f"Total Ratings: {row['Total_number_of_Ratings']}")
                    
elif option == "Collaborative Filtering":
    st.header("Book Recommendations via Collaborative Filtering")
    st.write("Select a book from the dropdown to see similar recommendations. Adjust the filters below to refine the results.")
    
    # käytetään pivot-taulukon indeksejä aka kirjojen nimiä
    book_list = pt.index.tolist()
    selected_book = st.selectbox("Select a Book:", book_list)
    
    # filtterit: average rating 0-6 ja total ratings 0-500
    st.sidebar.markdown("### Filter Recommendations")
    min_rating = st.sidebar.slider("Minimum Average Rating", min_value=0.0, max_value=6.0, value=0.0, step=0.1)
    min_total_ratings = st.sidebar.slider("Minimum Total Ratings", min_value=0, max_value=500, value=0, step=10)
    
    if st.button("Get Recommendations"):
        recs = recommend_cf_filtered(selected_book, min_rating, min_total_ratings)
        if recs is None or len(recs) == 0:
            st.error(f"No recommendations found for '{selected_book}' with the selected filters.")
        else:
            st.subheader(f"Books similar to **{selected_book}**:")
            for rec in recs:
                st.image(rec['Image-URL-M'], width=120)
                st.markdown(f"**{rec['Book-Title']}**")
                st.write(f"by {rec['Book-Author']}")
                st.write(f"Avg Rating: {rec['Average_Rating']} (Total Ratings: {rec['Total_number_of_Ratings']})")
                st.markdown("---")
