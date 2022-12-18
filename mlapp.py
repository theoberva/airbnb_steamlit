import pandas as pd
import streamlit as st
import numpy as np
import pydeck as pdk
from joblib import load

airbnb_image = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_B%C3%A9lo.svg/1200px-Airbnb_Logo_B%C3%A9lo.svg.png"
catboost_image = "https://camo.githubusercontent.com/978ad57e1fba31f89403bdc139b9dbaffe70d32e88e31e4017897d902955dcad/687474703a2f2f73746f726167652e6d64732e79616e6465782e6e65742f6765742d646576746f6f6c732d6f70656e736f757263652f3235303835342f636174626f6f73742d6c6f676f2e706e67"

st.sidebar.image(airbnb_image, caption='Project by Theo Bervanakis')
st.sidebar.info('''This project utilizes the CatBoost library devloped by Yandex. CatBoost is a gradient boosting framework that
                    excels at returning accurate and efficient models with categorical features.
                    ''')
st.sidebar.markdown('*Dataset: http://insideairbnb.com/get-the-data*')

st.sidebar.image(catboost_image)

@st.cache
def load_data():
    data = pd.read_parquet('cleaned_listings.gzip')
    return data

@st.cache
def load_vis_data():
    data = pd.read_parquet('cleaned_listings_streamlit.gzip')
    return data

    


df = load_data()
df_vis = load_vis_data()

model = load('best_model.pkl')

# initiate lists
locations = df['neighbourhood'].unique()
room_types = df['room_type'].unique()
property_types = df['property_type'].unique()
    

tab1, tab2 = st.tabs(["Explore", "Predict"])  


with tab1:
    st.title('Airbnb Listings in Melbourne :house_buildings:')
    """*Data last scrapped August 2022*"""




    max_accommodates = np.max(df_vis['accommodates'])

    min_price, max_price = st.slider('Price', value = [0,1500]) 
    location_vis = st.selectbox("Location ", sorted(locations), index=17)
    room_type_vis = st.selectbox('Room Type ', sorted(room_types))
    accommodates_vis = st.number_input('Guests', min_value = 1, max_value = max_accommodates)
    bedrooms_vis = st.number_input('Bedrooms ',0)
    bathrooms_vis =st.number_input('Bathrooms ', 0)


    df_filtered = df_vis.query("price >= @min_price and price <= @max_price and accommodates == @accommodates_vis and neighbourhood == @location_vis and bathrooms >= @bathrooms_vis and bedrooms >= @bedrooms_vis")



    midpoint = (np.average(df_filtered['latitude']), np.average(df_filtered["longitude"]))
    st.pydeck_chart(pdk.Deck(

        initial_view_state=pdk.ViewState(
            latitude= midpoint[0],
            longitude= midpoint[1],
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
               'HexagonLayer',
               data=df_filtered[['longitude', 'latitude']],
               get_position=['longitude', 'latitude'],
               radius=200,
               elevation_scale=10,
               elevation_range=[0, 1000],
               pickable=True,
               extruded=True,
               auto_highlight=True,
               coverage=1,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=df_filtered[['longitude', 'latitude']],
                get_position=['longitude', 'latitude'],
                get_color='[200, 30, 0, 160]',
                get_radius=200,
            ),
        ],
    ))

    if st.checkbox('Show Raw Data'):
        st.dataframe(df_filtered[['price','name', 'neighbourhood', 'property_type', 'accommodates','bathrooms', 'bedrooms','beds']])



with tab2:
    st.title('How Much Can You  Make On Airbnb?')

    # get user input
    location = st.selectbox("Location", sorted(locations))
    property_type =  st.selectbox('Property Type', sorted(property_types))
    room_type = st.selectbox('Room Type', sorted(room_types))
    accommodates = st.number_input('Accomodates', min_value = 1)
    bedrooms = st.number_input('Bedrooms', min_value = 0)
    beds =  st.number_input('Beds', min_value = 0)
    bathrooms =  st.number_input('Bathrooms', min_value = 0.0, step= 0.5)
    shared_bathroom_check = st.checkbox('Shared Bathroom')
    minimum_nights = st.number_input('Minimum Nights', min_value = 1)
    maximum_nights = st.number_input('Maximum Nights', min_value = 1)
    amenities = st.multiselect('Amenities', sorted(model.feature_names_[10:]))

    if shared_bathroom_check:
        shared_bathroom = 1
    else:
        shared_bathroom = 0




    # ml columns input
    features = model.feature_names_

    dict = {}
    def get_features():
        for feature in features[10:]:
            dict[feature] =  1 if feature in amenities else 0
        predict_amen_list = list(dict.values())
        results = result + predict_amen_list
        return results

    # predict price
    predict_price = st.button('Predict Price')

    if predict_price:
        result = [location, room_type, property_type, accommodates, bedrooms, beds, minimum_nights, maximum_nights, bathrooms, shared_bathroom]
        results = get_features()
        price = int(np.floor(model.predict(results)))
        st.title(f"${price} per night")
    










