from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
from func import rank_states
import plotly.express as px

app = FastAPI()
app.mount('/img', StaticFiles(directory="img"), name='img')

state_code_map = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# Load your dataset (replace with your actual path)
STATES = pd.read_csv("./data/compiled_data/predicted_states_data.csv")

# Allow only local access (adjust if needed)
origins = ["http://localhost", "http://127.0.0.1"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/api')
def api():
    return {'Status': 'Online'}

@app.get('/api/fetch_states')
def api_fetch_states():
    return {'States': list(STATES['State'].unique())}

@app.get('/api/fetch_columns')
def api_fetch_columns():
    return {'Columns': list(STATES.columns)}

@app.get('/api/rank_year')
def api_rank_year(year:int):
    if year >= 2014 and year <= 2030:
        data = rank_states(df=STATES, year=year)
        return data.to_dict(orient="records")
    else:
        return JSONResponse({'Error': 'Year param must be between 2014 and 2030 (inclusive, inclusive)'})

@app.get('/api/plot_feature')
def get_feature_plot(state: str = Query(...), feature: str = Query(...)):
    shading_starts = {
        'Population': 2021,
        'CO2 Emissions (Millions of tonnes)': 2024,
        'PPP': 2025, 
        'GDP': 2025,
        'Average Salary': 2025,
        'Average Disposable Income': 2025,
        'Violent Crimes Committed': 2024,
        'Unemployment Rate': 2023,
        'Hospital Count': 2023,
        'Traffic Fatalities': 2023,
        'Air Quality Index': 2025,
        'Hospitals Per Capita': 2023,
        'Violent Crimes Per Capita': 2024
    }

    df = STATES[STATES['State'] == state]
    if df.empty or feature not in df.columns:
        return {"error": "Invalid state or feature"}

    df_grouped = df.groupby('Year').mean(numeric_only=True).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_grouped['Year'], df_grouped[feature], marker='o', label=feature)
    ax.set_title(f"{feature} in {state}")
    ax.set_xlabel("Year")
    ax.set_ylabel(feature)
    ax.grid(True)

    if feature in shading_starts:
        start = shading_starts[feature]
        df_future = df_grouped[df_grouped['Year'] >= start]
        if not df_future.empty:
            ax.axvspan(df_future['Year'].iloc[0], df_grouped['Year'].iloc[-1],
                       color='orange', alpha=0.3, label='Predicted Data')

    ax.legend(loc='upper left')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.read(), media_type="image/png")

@app.get('/api/generate_rankings_map')
def root_api_generate_rankings_map(year: int):
    try:
        # Copy the data for the selected year
        df_year = STATES[STATES["Year"] == year].copy()

        if df_year.empty:
            return {"error": f"No data found for year {year}"}

        # Define scoring columns based on your data
        scoring_columns = list(STATES.columns)[3:]
        missing_cols = [col for col in scoring_columns if col not in df_year.columns]
        if missing_cols:
            return {"error": f"Missing columns: {missing_cols}"}

        # Calculate the final score (sum of the scoring columns)
        df_year["Score"] = df_year[scoring_columns].sum(axis=1)

        # Normalize the final score to a 1-50 ranking (lower scores are better)
        df_year["Rank"] = pd.qcut(df_year["Score"], 50, labels=False) + 1  # 1 is the best rank, 50 is the worst

        # Map the state codes (assuming you have a map for state codes)
        df_year["State Code"] = df_year["State"].map(state_code_map)
        df_year = df_year.dropna(subset=["State Code"])

        # Create the choropleth map using the "Rank" for coloring
        fig = px.choropleth(
            df_year,
            locations="State Code",
            locationmode="USA-states",
            color="Rank",
            color_continuous_scale="Viridis_r",  # Use a reversed scale if you want the lowest rank as green
            scope="usa",
            labels={"Rank": "Rank (1 = Best)"},
            title=f"{year} Projected State Rankings"
        )

        # Save the figure to a BytesIO buffer as a PNG image
        buf = BytesIO()
        fig.write_image(buf, format="png")
        buf.seek(0)

        # Return the image in the response
        return Response(content=buf.read(), media_type="image/png")

    except Exception as e:
        return {"error": str(e)}

@app.get('/api/dl_predicted_states_data')
def root_api_download_predicted_states_data():
    try:
        return FileResponse('./data/compiled_data/predicted_states_data.csv', media_type='text/csv', filename='predicted_states_data.csv')
    except:
        return {'DownloadError': 'Something went wrong when downloading this file.'}
    
@app.get('/api/dl_unpredicted_states_data')
def root_api_download_unpredicted_states_data():
    try:
        return FileResponse('./data/compiled_data/unpredicted_states_data.csv', media_type='text/csv', filename='unpredicted_states_data.csv')
    except:
        return {'DownloadError': 'Something went wrong when downloading this file.'}