from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
from func import rank_states

app = FastAPI()
app.mount('/img', StaticFiles(directory="img"), name='img')

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