import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
from os import getenv
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()

st.set_page_config(page_title="State Rankings", layout="wide")

st.sidebar.title("🗺️ Navigation")
page = st.sidebar.radio('Pages:', ["Introduction", "The Data", "Methodology", "Findings", "API", "Information"])

match page:
    case "Introduction":
        st.image('./img/banner_art.jpg', width=None)
        st.title("📊 State Rankings by Conditions")
        st.write(
            """
            What States are the best to live in, based on Data.

            By: David J Chester
            """
        )

        st.subheader('Objectives')
        st.write("""The goal of this analysis and its predictions is to be able to say, with confidence, what State will be the best to live in, in the year 2030. Many factors are at play, as seen in \"The Data\".
                \nWe will determine this by ranking the states per year, 1 through 50, for each column. Then at the end we will add everyones rankings across the board, and the lowest number is the Best State to live in.
                """)
    case "The Data":
        st.title("🔬 The Data")
        st.write("""
    This project leverages data on various factors that influence the quality of life across different states in the USA.\n
    Aiming to predict the best states to live in, in future years, based on metrics like population, CO2 emissions, crime rates, 
    Hospital Counts, GDP, and more. Below is an overview of the dataset used for analysis:
    """)
        
        st.subheader("Columns in the Dataset")
        st.write("""
        The dataset contains multiple columns that track various metrics for each state over different years. This Data was constructed from 96 files, as well as online data made unavailable for download, in which case data was manually re-entered to fulfill data requirements.
        \n
        **Data Columns**:
        - **State**: The State name.
        - **Year**: The year for which the data is recorded.
        - **Population**: The population of the state for that year. (90% of all population data has been predicted as the US Census is only every 10 years)
        - **CO2 Emissions (Millions of tonnes)**: Annual CO2 emissions for the state in millions of tonnes. (Transportational, Industrial, Personal)
        - **PPP**: The state\'s Price Parity (How much $100 is worth in the State)
        - **GDP**: The state\'s Gross Domestic Product.
        - **Average Salary**: The average salary in the state.
        - **Average Disposable Income**: The amount of money after taxes.
        - **Violent Crimes Committed**: Violent crimes per year (Rape, Homicide/NonNegligent Manslaughter, Robbery, Aggravated Assault).
        - **Unemployment Rate**: Percentage of people unemployed in the state.
        - **Hospital Count**: Number of hospitals in the state.
        - **Traffic Fatalities**: The amount of traffic fatalites per 100,000 people (per capita)
        - **Air Quality Index**: The level of air pollution in the state.
        - **Violent Crimes Per Capita**: The amount of violent crimes for every 100,000 people (per capita)
        - **Hospitals Per Capita**: The amount of hospitals available for every 100,000 people (per capita).
        """)

    case "Methodology":
        st.title('🧪 Methodology')
        st.write("""This data was assembled from 94 seperate files with each column being manually sorted, updated and modified to fit""")
        st.subheader('Populations')
        st.write("""First, the population had to be predicted as this would be the backbone to all of our other predictions. The United States Census is conducted at the start of every decade, which means 90% of all years did not have a population associated.
                \nNote: Blue means the population has decreased from 2020 to 2030""")
        st.image(image='./img/predicted_pops_2020_vs_2030.png', width=1000)

        st.subheader('Predictions')
        st.write("""All of our predictions use the same polynomial format, and required hours of tweaking to get a reasonable set of trends, some states had to be manually adjusted to avoid extremely outrageous outputs. Two columns do not use the same model:
- **Hospital Count**: The polynomial provided extreme drops and rises in Hospital Counts as well as predicting partial values for the hospitals (For example, we cant have 120.3462 hospitals) 
- **Violent Crimes Committed**: This column needed other items factored in: Population, Disposable Income (Criminality is tied to Poverty), Unemployment Rate (Higher unemployment rate = higher crime rate)""")
        
        st.subheader("Explore the Data")
        states = requests.get(f'http://{getenv("address")}/api/fetch_states')
        columns = requests.get(f'http://{getenv("address")}/api/fetch_columns')
        state = st.selectbox("Select a state to explore:", states.json()['States'])
        column = st.selectbox("Select the column to have displayed", columns.json()['Columns'][1:])
        if st.button('Generate Image'):
            res = requests.get(f'http://{getenv("address")}/api/plot_feature',params={'state': state, 'feature': column})
            if res.status_code == 200:
                img = Image.open(BytesIO(res.content))
                st.image(img, caption=f'{column} in {state} over time.', use_container_width=True)
            else:
                st.write('Something went wrong when generating. Please try a different selection.')
        
    case "Findings":
        st.title('🔍 Findings')
        st.write('After predicting all of our data using Linear Regression, and Polynomial Regression, I determined what would be done moving forward for each column, and created new columns to make some data more fair between states, for example, hospitals per capita, as populations would determine if more or less hospitals are necessary.')
        st.write('I then took all the rows and started converting their data to ratings for that specific data-type, 1-50, afterwards I summed the entire state data and the lowest is the winner.')
        
        
        st.subheader("Predicted State Rankings")
        year = st.number_input("Select Year", min_value=2014, max_value=2030, value=2030)
        api_url = f"http://{getenv('address')}/api/generate_rankings_map?year={year}"
        response = requests.get(api_url)
        if response.status_code == 200:
            try:
                img = Image.open(BytesIO(response.content))
                st.image(img)
            except Exception as e:
                st.error(f"Error rendering the image: {str(e)}")

        if 'data_visible' not in st.session_state:
            st.session_state['data_visible'] = False
        if st.button('Toggle Raw Rank Data'):
            # Toggle the visibility state
            st.session_state['data_visible'] = not st.session_state['data_visible']
            
            if st.session_state['data_visible']:
                # Fetch the data from the API when the button is clicked and toggled
                response = requests.get(f'http://{getenv("address")}/api/rank_year?year={year}')
                if response.status_code == 200:
                    data = response.json()
                    data = pd.DataFrame(data)
                    st.dataframe(data=data)
                else:
                    st.error('Failed to obtain the 2030 State Ranking Data from the API.')

    case "API":
        st.title('🤖 API Documentation')
        st.write('All requests are `GET` requests. There are no api_keys or request_limits.')

        st.write(f'`http://{getenv("address")}/api` - Test endpoint to confirm if the api is active')
        st.write(f'`http://{getenv("address")}/api/fetch_columns` - Returns json of each column in the predicted dataset')
        st.write(f'`http://{getenv("address")}/api/fetch_states` - Returns json of each state in the predicted dataset')
        st.write(f'`http://{getenv("address")}/api/rank_year?year=2017` - Returns a jsonified dataframe with the rankings for each state. Year param must be between 2014 and 2030 (inclusive, inclusive)')
        st.write(f'`http://{getenv("address")}/api/plot_feature?state=Alabama&feature=Population` - Returns a pyplot image showcasing the States Feature over time for all data in the predicted dataset')
        st.write(f'`http://{getenv("address")}/api/dl_predicted_states_data` - Download link for the predicted states dataset (not the ranked data)')
        st.write(f'`http://{getenv("address")}/api/dl_unpredicted_states_data` - Download link for the non predicted states dataset (for making your own models) (not the ranked data)')
        st.write(f'`http://{getenv("address")}/api/generate_rankings_map?year=2030` - View a Map of the United States color coded for rankings. Year param must be between 2014 and 2030 (inclusive, inclusive)')

    case "Information":
        st.title("ℹ️ Information")
        st.write("This website is hosted via `databasemart.com` on a 2 core/4gb RAM/60GB SSD Linux Ubuntu Server, and is not endorsed, nor approved by `Database Mart LLC`")
        st.write("The creator of this site instantiated this Linux Server as a Final Project Demonstration Environment. This server WILL BE SCRUBBED.")
        st.write("All data for this server may be found at the following github repository: https://github.com/AviAlibi/Final-Project")