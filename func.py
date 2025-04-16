import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Tuple
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import seaborn as sns
from math import floor
import zipfile
import requests
from io import BytesIO
from time import sleep

def clean_usa_from_excel():
    '''pulls data from table1.xlsx and puts in in /data/modified/state_emissions_per_year.csv for use'''
    data = pd.read_excel(io='./data/original/table1.xlsx',
                         skiprows=4, sheet_name='Table 1')
    columns = ['State']
    for year in range(1970, 2023):
        columns.append(year)
    data = data[columns]
    data = data[data['State'].isin(data['State'].unique()[0:51])]
    data.to_csv('./data/modified/state_emissions_per_year.csv')
    return None

def prepare_data() -> pd.DataFrame:
    clean_usa_from_excel()
    ecie = pd.read_csv('./data/custom/energy_consumption_in_exojoules.csv')
    ecie = pd.melt(ecie, id_vars=['Country'], var_name='Year',value_name='Energy Consumption (EJ)')
    ecpc = pd.read_csv('./data/custom/energy_consumption_per_capita.csv')
    ecpc = pd.melt(ecpc, id_vars=['Country'], var_name='Year',value_name='Consumption Per Capita (GJ)')
    co2mt = pd.read_csv('./data/custom/co2_million_tonnes.csv')
    co2mt = pd.melt(co2mt, id_vars=['Country'], var_name='Year',value_name='CO2 Emissions (Millions of Tons)')
    ENERGY = pd.merge(ecie, ecpc, on=['Country', 'Year'], how='inner')
    ENERGY = pd.merge(ENERGY, co2mt, on=['Country', 'Year'], how='inner')
    ENERGY['Year'] = ENERGY['Year'].astype(int)


    POPULATION = pd.read_csv('./data/modified/modified_population.csv')
    POPULATION = conv_pop_dict_to_pop_df(population_dict=predict_populations_1970_2030(POPULATION=POPULATION))
    
    DATA = pd.merge(POPULATION, ENERGY, on=['Country', 'Year'], how='outer')

    DATA = DATA[DATA['Year'] >= 1970]
    DATA['Year'] = DATA['Year'].astype(int)

    return DATA

def predict_populations_1970_2030(POPULATION: pd.DataFrame) -> dict:
    countries = POPULATION['Country'].unique()
    container = {}
    for country in countries:
        country_data = {str(year): None for year in range(
            1970, 2031)}
        temp = POPULATION[POPULATION['Country'] == country]
        melted = temp.melt(id_vars=['Country', 'Continent', 'Area (km²)', 'Density (per km²)'],
                        value_vars=[
                            '1970 Population', '1980 Population', '1990 Population',
                            '2000 Population', '2010 Population', '2015 Population',
                            '2020 Population', '2022 Population'
        ],
            var_name='Year', value_name='Population')
        melted['Year'] = melted['Year'].str.extract(r'(\d+)').astype(str)
        for _, row in melted.iterrows():
            year = row['Year']
            population = row['Population']
            country_data[year] = population
        container[country] = country_data
        years = list(country_data.keys())
        valid_years = [int(year)
                        for year in years if country_data[year] is not None]
        for i in range(len(valid_years) - 1):
            start_year = valid_years[i]
            end_year = valid_years[i + 1]
            start_pop = country_data[str(start_year)]
            end_pop = country_data[str(end_year)]
            growth = end_pop - start_pop
            years_between = end_year - start_year
            for j in range(1, years_between):
                year_to_fill = start_year + j
                country_data[str(year_to_fill)] = floor(
                    start_pop + (growth / years_between) * j)
        country_data.update(predict_future_population(
            country_data=country_data))
        container[country] = country_data
    return container

def predict_future_population(country_data: dict, degree=3) -> dict:
    # Extract years and populations from the dictionary
    years = np.array([int(year) for year in country_data.keys()
                     if country_data[year] is not None]).reshape(-1, 1)
    populations = np.array([country_data[year] for year in country_data.keys(
    ) if country_data[year] is not None]).reshape(-1, 1)

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(years)

    # Create and fit the model
    model = LinearRegression()
    model.fit(X_poly, populations)

    # Prepare the years for prediction (2023 to 2030)
    future_years = np.array(range(2023, 2031)).reshape(-1, 1)
    future_X_poly = poly.transform(future_years)

    # Predict populations for the future years
    future_populations = model.predict(future_X_poly)

    # Store the predictions in a dictionary
    predictions = {str(year[0]): round(pop[0])
                   for year, pop in zip(future_years, future_populations)}

    return predictions

def format_population(x, _):
    if x >= 1_000_000_000:  # Billions
        num = x / 1_000_000_000
        if num == 1.0:
            num = 1
        return f'{num:.1f} B'  # Converts to billions
    elif x >= 1_000_000:  # Millions
        return f'{int(x / 1_000_000):,} M'  # Converts to millions
    else:  # Less than a million
        return f'{int(x):,}'  # Show the number as is

def render_population_predictions(DATA:pd.DataFrame) -> None:
    sns.set(style='whitegrid')

    # Create a line plot
    plt.figure(figsize=(7, 5))

    # Loop through each country and plot its population over the years
    for country in DATA['Country'].unique():
        country_data = DATA[DATA['Country'] == country]
        # No marker specified for solid line
        plt.plot(
            country_data['Year'],
            country_data['Population'],
            label=country, linewidth=2, alpha=0.8)

    # Add titles and labels
    plt.axvspan(2023, 2030, color='red', alpha=0.1, label='Predicted Data')
    plt.title('Predicted Population of Countries (1970-2030)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Predicted Population', fontsize=14)
    plt.legend(title='Country', bbox_to_anchor=(
        1.05, 1), loc='upper left', fontsize=10)
    plt.xticks(rotation=45)  # Rotate x-ticks for better readability
    plt.gca().yaxis.set_major_formatter(formatter=format_population)
    plt.tight_layout()
    plt.grid(visible=True)
    plt.savefig('./img/predicted_populations_1970_2030.png')

def conv_pop_dict_to_pop_df(population_dict:dict) -> pd.DataFrame:
    columns = ['Country', 'Year', 'Population']
    data = pd.DataFrame(columns=columns)
    for key, value in population_dict.items():
        for year, population in value.items():
            new_row = pd.DataFrame([{'Country': key, 'Year': year, 'Population': population}])
            data = pd.concat([data, new_row], ignore_index=True)
    data['Year'] = pd.to_numeric(data['Year'])
    return data

def predict_states_populations_1960_2030(States_Pops: pd.DataFrame, state_degrees: dict) -> dict:
    states = States_Pops['Name'].unique()
    container = {}
    for state in states:
        state_data = {str(year): None for year in range(1960, 2031)}
        temp = States_Pops[States_Pops['Name'] == state]

        melted = temp.melt(
            id_vars=['Name'],
            value_vars=['1960', '1970', '1980',
                        '1990', '2000', '2010', '2020'],
            var_name='Year',
            value_name='Population'
        )

        for _, row in melted.iterrows():
            year = row['Year']
            population = row['Population']
            state_data[year] = population

        years = list(state_data.keys())
        valid_years = [int(year)
                       for year in years if pd.notna(state_data[year])]

        for i in range(len(valid_years) - 1):
            start_year = valid_years[i]
            end_year = valid_years[i + 1]
            start_pop = state_data[str(start_year)]
            end_pop = state_data[str(end_year)]

            if pd.isna(start_pop) or pd.isna(end_pop):
                continue

            growth = end_pop - start_pop
            years_between = end_year - start_year
            for j in range(1, years_between):
                year_to_fill = start_year + j
                state_data[str(year_to_fill)] = floor(
                    start_pop + (growth / years_between) * j)

        # Get the degree for this state, default to 3 if not specified
        degree = state_degrees.get(state, 3)
        state_data.update(predict_future_state_population(state_data, degree))
        container[state] = state_data

    return container

def predict_future_state_population(state_data: dict, degree=3) -> dict:
    # Extract years and populations, filtering out NaNs
    years, populations = zip(*[
        (int(year), population)
        for year, population in state_data.items()
        if population is not None and not pd.isna(population)
    ])

    # Convert to numpy arrays
    years = np.array(years).reshape(-1, 1)
    populations = np.array(populations).reshape(-1, 1)

    # If there aren't enough points to fit the model, skip prediction
    if len(years) < degree + 1:
        return {}

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(years)

    # Create and fit the model
    model = LinearRegression()
    model.fit(X_poly, populations)

    # Predict for future years
    future_years = np.array(range(2021, 2031)).reshape(-1, 1)
    future_X_poly = poly.transform(future_years)
    future_populations = model.predict(future_X_poly)

    # Store predictions
    predictions = {str(year[0]): round(pop[0])
                   for year, pop in zip(future_years, future_populations)}

    return predictions

def convert_state_pop_dict_to_df(state_dict: dict) -> pd.DataFrame:
    records = []
    for state, yearly_data in state_dict.items():
        for year, pop in yearly_data.items():
            records.append(
                {'State': state, 'Year': int(year), 'Population': pop})
    return pd.DataFrame(records)

def render_energy_predictions(DATA: pd.DataFrame) -> None:
    sns.set_theme(style='whitegrid')

    columns = [
        'Energy Consumption (EJ)',
        'Consumption Per Capita (GJ)',
        'CO2 Emissions (Millions of Tons)'
    ]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 14), sharex=True)

    for i, column in enumerate(columns):
        ax = axes[i]
        for country in DATA['Country'].unique():
            country_data = DATA[DATA['Country'] == country]

            ax.plot(
                country_data['Year'],
                country_data[column],
                label=country, linewidth=2, alpha=0.8
            )

        ax.axvspan(2023, 2030, color='red', alpha=0.1,
                   label='Predicted Data' if i == 0 else "")
        ax.set_title(
            f'Predicted {column} of Countries (2010-2030)', fontsize=14)
        ax.set_ylabel(column, fontsize=12)
        ax.grid(True)

        if i == 0:
            ax.legend(title='Country', bbox_to_anchor=(
                1.05, 1), loc='upper left', fontsize=9)

    axes[-1].set_xlabel('Year', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./img/predicted_all_metrics.png')

def predict_energy_data_global(DATA: pd.DataFrame, degree_mapping: dict) -> pd.DataFrame:
    '''Reads in all the data and predicts the energy usage, emissions, and per capita usage based on trends.'''

    # Ensure 'Population' is numeric, converting if necessary
    DATA['Population'] = pd.to_numeric(DATA['Population'], errors='coerce')

    # Filter data to include only the 2010s and beyond
    DATA_2010s = DATA[DATA['Year'] >= 2000]

    # Iterate over each country in the degree mapping
    for country in degree_mapping.keys():
        country_data = DATA_2010s[DATA_2010s['Country'] == country]

        # Prepare years and relevant columns
        years = country_data['Year'].values.reshape(-1, 1)
        populations = country_data['Population'].values.reshape(-1, 1)

        # Get the polynomial degree for the current country
        degree = degree_mapping[country]

        # Check if there are enough data points to fit a model
        if len(years) >= 2 and np.any(~np.isnan(populations)):
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(years)

            # Fit the population model
            population_model = LinearRegression()
            population_model.fit(X_poly, populations)

            # Predict future populations for the next years (2024 to 2030)
            future_years = np.array(range(2024, 2031)).reshape(-1, 1)
            future_X_poly = poly.transform(future_years)
            future_populations = population_model.predict(future_X_poly)

            # Store predicted populations back into DATA
            for year, pop in zip(future_years.flatten(), future_populations.flatten()):
                DATA.loc[(DATA['Country'] == country) & (
                    DATA['Year'] == year), 'Population'] = round(pop)

            # Fit models for energy consumption, per capita consumption, and CO2 emissions
            for target_col in ['Energy Consumption (EJ)', 'Consumption Per Capita (GJ)', 'CO2 Emissions (Millions of Tons)']:
                y_target = country_data[target_col].values.reshape(-1, 1)

                # Check for non-NaN values in y_target
                valid_mask = ~np.isnan(y_target).flatten()

                # Ensure we have at least two valid data points
                if len(years) >= 2 and np.any(valid_mask):
                    # Fit the model using valid indices
                    model = LinearRegression()
                    model.fit(X_poly[valid_mask], y_target[valid_mask])

                    # Predict for future years
                    future_values = model.predict(future_X_poly)

                    # Update the DATA DataFrame with predictions
                    for year, value in zip(future_years.flatten(), future_values.flatten()):
                        DATA.loc[(DATA['Country'] == country) & (
                            DATA['Year'] == year), target_col] = round(value)

    return DATA

def render_states_barh_overlay(states_df: pd.DataFrame):
    # Filter for 2020 and 2030, align both by state
    df_2020 = states_df[states_df['Year'] == 2020].copy()
    df_2030 = states_df[states_df['Year'] == 2030].copy()

    # Ensure same order of states
    df_2020.sort_values('State', inplace=True)
    df_2030 = df_2030.set_index('State').loc[df_2020['State']].reset_index()

    fig, ax = plt.subplots(figsize=(16, 14))

    y_positions = range(len(df_2020))

    # Plot 2020 populations
    ax.barh(y_positions, df_2020['Population'],
            label='2020', color='blue', alpha=0.5)

    # Plot 2030 populations
    ax.barh(y_positions, df_2030['Population'],
            label='2030 (Projected)', color='red', alpha=0.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_2020['State'])
    ax.invert_yaxis()  # Optional: states go top-to-bottom

    ax.set_title('State Populations: 2020 vs 2030 (Projected)', fontsize=18)
    ax.set_xlabel('Population')
    ax.set_ylabel('State')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    # Format X-axis
    def millions_or_billions(x, _):
        if x >= 1_000_000_000:
            return f'{x/1e9:.1f}B'
        return f'{x/1e6:.0f}M'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(millions_or_billions))
    ax.legend()

    plt.tight_layout()
    plt.savefig('./img/predicted_pops_2020_vs_2030.png')

def state_data_assembly(STATES:pd.DataFrame) -> pd.DataFrame:
    '''Takes in STATES:pd.DataFrame and adds on all its acccessory columns, does not make any predictions for any of this data'''
    STATES = emissions_constructor(STATES=STATES)
    STATES = price_parities_constructor(STATES=STATES)
    STATES = gdp_constructor(STATES=STATES)
    STATES = avg_salary_constructor(STATES=STATES)
    STATES = disposable_income_constructor(STATES=STATES)
    STATES = violent_crimes_constructor(STATES=STATES)
    STATES = unemployment_constructor(STATES=STATES)
    STATES = hospitals_constructor(STATES=STATES)
    STATES = traffic_fatalities_constructor(STATES=STATES)
    STATES = air_quality_constructor(STATES=STATES)
    STATES.to_csv('./data/compiled_data/states.csv')
    return STATES

def emissions_constructor(STATES:pd.DataFrame) -> pd.DataFrame:
    States_Emissions = pd.read_csv(
        './data/modified/state_emissions_per_year.csv')
    States_Emissions = States_Emissions.iloc[:, 1:]
    States_Emissions = pd.melt(States_Emissions, id_vars=['State'], var_name='Year',
                            value_name='CO2 Emissions (Millions of tonnes)')
    States_Emissions['Year'] = States_Emissions['Year'].astype(int)
    STATES = pd.merge(STATES, States_Emissions, on=['State', 'Year'], how='left')
    return STATES

def price_parities_constructor(STATES:pd.DataFrame) -> pd.DataFrame:
    States_Price_Parities = pd.read_csv('./data/modified/price_parities.csv')
    States_Price_Parities = States_Price_Parities.iloc[:, 1:]
    States_Price_Parities.rename(columns={'GeoName': 'State'}, inplace=True)
    States_Price_Parities = pd.melt(States_Price_Parities, id_vars=['State'], var_name='Year',
                                    value_name='PPP')
    States_Price_Parities['Year'] = States_Price_Parities['Year'].astype(int)
    STATES = pd.merge(STATES, States_Price_Parities,
                    on=['State', 'Year'], how='left')
    return STATES

def gdp_constructor(STATES:pd.DataFrame) -> pd.DataFrame:
    States_GDP = pd.read_csv('./data/modified/state_gdp.csv')
    States_GDP = States_GDP.iloc[:, 1:]
    States_GDP.rename(columns={'GeoName': 'State'}, inplace=True)
    States_GDP = pd.melt(States_GDP, id_vars=['State'], var_name='Year',
                        value_name='GDP')
    States_GDP['Year'] = States_GDP['Year'].astype(int)
    STATES = pd.merge(STATES, States_GDP,
                    on=['State', 'Year'], how='left')
    return STATES

def avg_salary_constructor(STATES: pd.DataFrame) -> pd.DataFrame:
    States_AVG_Salary = pd.read_csv('./data/modified/state_average_salary.csv')
    States_AVG_Salary = States_AVG_Salary.iloc[:, 1:]
    States_AVG_Salary.rename(columns={'GeoName': 'State'}, inplace=True)
    States_AVG_Salary = pd.melt(States_AVG_Salary, id_vars=['State'], var_name='Year',
                                value_name='Average Salary')
    States_AVG_Salary['Year'] = States_AVG_Salary['Year'].astype(int)
    STATES = pd.merge(STATES, States_AVG_Salary,
                    on=['State', 'Year'], how='left')
    return STATES

def disposable_income_constructor(STATES: pd.DataFrame) -> pd.DataFrame:
    States_Disposable_Income = pd.read_csv(
        './data/modified/state_disposable_income.csv')
    States_Disposable_Income = States_Disposable_Income.iloc[:, 1:]
    States_Disposable_Income.rename(columns={'GeoName': 'State'}, inplace=True)
    States_Disposable_Income = pd.melt(States_Disposable_Income, id_vars=['State'], var_name='Year',
                                    value_name='Average Disposable Income')
    States_Disposable_Income['Year'] = States_Disposable_Income['Year'].astype(int)
    STATES = pd.merge(STATES, States_Disposable_Income,
                    on=['State', 'Year'], how='left')
    return STATES

def violent_crimes_constructor(STATES: pd.DataFrame) -> pd.DataFrame:
    States_VC = pd.read_csv('./data/crime_stats/Violent Crimes.csv')
    States_VC = pd.melt(States_VC, id_vars=['State'], var_name='Year',
                        value_name='Violent Crimes Committed')
    States_VC['Year'] = States_VC['Year'].astype(int)
    STATES = pd.merge(STATES, States_VC,
                    on=['State', 'Year'], how='left')
    return STATES

def unemployment_constructor(STATES:pd.DataFrame) -> pd.DataFrame:
    unemployment = pd.read_csv(
        './data/original/Unemployment in America Per US State.csv')
    unemployment = unemployment[unemployment['State/Area']
                                .isin(STATES['State'].unique())]
    unemployment = unemployment.groupby(
        ['State/Area', 'Year']).mean(numeric_only=True).reset_index()
    unemployment = unemployment.round(2)
    unemployment = unemployment[['State/Area', 'Year',
                                'Percent (%) of Labor Force Unemployed in State/Area']]
    unemployment.rename(columns={'State/Area': 'State',
                        'Percent (%) of Labor Force Unemployed in State/Area': 'Unemployment Rate'}, inplace=True)
    unemployment[unemployment['State'] == 'North Carolina']
    STATES = pd.merge(STATES, unemployment, on=['State', 'Year'], how='left')
    return STATES

def hospitals_constructor(STATES:pd.DataFrame) -> pd.DataFrame:
    hospitals = pd.DataFrame(columns=['State'])
    years = range(1999, 2023)
    for year in years:
        temp = pd.read_csv(
            f'./data/original/hospital_data/hospitals_{year}.csv', index_col=False)
        temp.rename(columns={'Total Hospitals': f'{year}',
                    'Location': 'State'}, inplace=True)
        hospitals = pd.merge(hospitals, temp, on=['State'], how='outer')
    hospitals
    hospitals = pd.melt(hospitals, id_vars=['State'], var_name='Year',
                        value_name='Hospital Count')
    hospitals['Year'] = hospitals['Year'].astype(int)
    STATES = pd.merge(STATES, hospitals,
                    on=['State', 'Year'], how='left')
    return STATES

def traffic_fatalities_constructor(STATES:pd.DataFrame) -> pd.DataFrame:
    traffic_fatalities = pd.read_csv(
        './data/modified/traffic_fatalities_per_capita.csv')
    STATES = pd.merge(STATES, traffic_fatalities, on=['State', 'Year'], how='left')
    return STATES

def air_quality_constructor(STATES: pd.DataFrame) -> pd.DataFrame:
    records = []
    for year in range(1980, 2025):
        temp = pd.read_csv(
            f'./data/original/air_quality_data/annual_aqi_by_county_{year}.csv', index_col=False)
        air_quality = temp.groupby('State')['Median AQI'].mean().reset_index()
        air_quality['Year'] = year
        records.append(air_quality)

    air_quality = pd.concat(records, ignore_index=True)
    air_quality = air_quality[['State', 'Year', 'Median AQI']]
    air_quality.rename(
        columns={'Median AQI': 'Air Quality Index'}, inplace=True)
    STATES = pd.merge(STATES, air_quality, on=['State', 'Year'], how='left')
    return STATES

def violent_crimes_per_capita_constructor(STATES:pd.DataFrame) -> pd.DataFrame:
    STATES['Violent Crime Rate (per capita)'] = (
        STATES['Violent Crimes Committed'] / STATES['Population'])
    return STATES

def hospitals_per_capita_constructor(STATES:pd.DataFrame) -> pd.DataFrame:
    STATES['Hospitals Per Capita'] = (
        STATES['Hospital Count'] / STATES['Population'])
    return STATES

def get_all_aqi_files():
    years = range(1980, 2025)
    for year in years:
        try:
            url = f'https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_{year}.zip'
            response = requests.get(url)
            zip_file = zipfile.ZipFile(BytesIO(response.content))
            zip_file.extract(member=f'annual_aqi_by_county_{year}.csv', path='./data/original/air_quality_data')
            print(f'{year}: Success')
            sleep(1)
        except:
            print(f'{year}: FAIL FAIL FAIL')

def predict_missing_column_by_state(df, column, degree_dict=None, default_degree=2, forecast_year=2030):
    df = df.copy()
    states = df['State'].unique()

    # Ensure the column exists in the dataframe
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    for state in states:
        state_df = df[df['State'] == state].sort_values('Year')

        # Skip if the column is entirely NaN for the given state
        if state_df[column].dropna().empty:
            continue

        # Find the last year with data
        last_valid_idx = state_df[column].last_valid_index()
        last_valid_year = state_df.loc[last_valid_idx, 'Year']

        # Define years to predict
        pred_years = np.arange(last_valid_year + 1, forecast_year + 1)
        if len(pred_years) == 0:
            continue

        # Fit polynomial on existing data
        existing = state_df.dropna(subset=[column])
        X_train = existing['Year'].values.reshape(-1, 1)
        y_train = existing[column].values

        degree = degree_dict.get(state, default_degree) if degree_dict else default_degree
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X_train)

        model = LinearRegression()
        model.fit(X_poly, y_train)

        # Predict
        X_pred = poly.transform(pred_years.reshape(-1, 1))
        y_pred = model.predict(X_pred)

        # Insert predictions back into the DataFrame
        for year, value in zip(pred_years, y_pred):
            mask = (df['State'] == state) & (df['Year'] == year)
            if not df[mask].empty:
                df.loc[mask, column] = value
            else:
                # Create new row if year didn't exist yet
                new_row = {
                    'State': state,
                    'Year': year,
                    'Population': np.nan,  # or estimate separately
                    column: value
                }
                df = pd.concat([df, pd.DataFrame([new_row])],
                                ignore_index=True)

    # Re-sort the full dataframe
    df = df.sort_values(['State', 'Year']).reset_index(drop=True)
    return df

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def predict_hospital_counts(
    df,
    forecast_year=2030,
    clamp_range=(0, 500)
):
    df = df.copy()
    column = 'Hospital Count'
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    states = df['State'].unique()

    for state in states:
        state_df = df[df['State'] == state].sort_values('Year')

        if state_df[column].dropna().empty:
            continue

        last_valid_idx = state_df[column].last_valid_index()
        last_valid_year = state_df.loc[last_valid_idx, 'Year']
        pred_years = np.arange(last_valid_year + 1, forecast_year + 1)
        if len(pred_years) == 0:
            continue

        existing = state_df.dropna(subset=[column])
        X_train = existing['Year'].values.reshape(-1, 1)
        y_train = existing[column].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        X_pred = pred_years.reshape(-1, 1)
        y_pred = model.predict(X_pred)
        y_pred = np.clip(y_pred, *clamp_range)

        # Round predicted values to the nearest integer
        y_pred = np.round(y_pred).astype(int)

        for year, value in zip(pred_years, y_pred):
            mask = (df['State'] == state) & (df['Year'] == year)
            if not df[mask].empty:
                df.loc[mask, column] = value
            else:
                df = pd.concat([df, pd.DataFrame([{
                    'State': state,
                    'Year': year,
                    'Population': np.nan,
                    column: value
                }])], ignore_index=True)

    return df.sort_values(['State', 'Year']).reset_index(drop=True)

def predict_violent_crime_time_aware(
    df,
    forecast_year=2030,
    features=['Population', 'Unemployment Rate', 'Average Disposable Income', 'GDP'],
    target='Violent Crimes Committed',
    max_yearly_change=0.10  # allow up to 10% change per year
):
    df = df.copy()
    states = df['State'].unique()

    for state in states:
        state_df = df[df['State'] == state].sort_values('Year').reset_index(drop=True)

        if state_df[target].dropna().empty:
            continue

        state_df['Lag_Crime'] = state_df[target].shift(1)

        for idx, row in state_df.iterrows():
            df_idx = df[(df['State'] == state) & (df['Year'] == row['Year'])].index
            if not df_idx.empty:
                df.loc[df_idx[0], 'Lag_Crime'] = row['Lag_Crime']

        used_features = features + ['Year', 'Lag_Crime']

        train_df = df[(df['State'] == state) & df[used_features + [target]].notnull().all(axis=1)]
        if train_df.empty:
            continue

        X_train = train_df[used_features].values
        y_train = train_df[target].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        last_year = df[(df['State'] == state) & df[target].notna()]['Year'].max()
        pred_years = np.arange(last_year + 1, forecast_year + 1)

        last_known_crime = df[(df['State'] == state) & (df['Year'] == last_year)][target].values[0]

        for year in pred_years:
            prev_row = df[(df['State'] == state) & (df['Year'] == year - 1)]
            if prev_row.empty:
                continue

            new_row = {
                'State': state,
                'Year': year,
                'Lag_Crime': last_known_crime
            }

            for col in features:
                val = prev_row[col].values[0]
                new_row[col] = val

            if any(pd.isna(new_row[col]) for col in new_row):
                continue

            X_pred = pd.DataFrame([new_row])[used_features].values
            raw_pred = model.predict(X_pred)[0]

            # Clamp prediction to within a % range of previous year
            upper_bound = last_known_crime * (1 + max_yearly_change)
            lower_bound = last_known_crime * (1 - max_yearly_change)
            prediction = np.clip(raw_pred, lower_bound, upper_bound)
            prediction = max(0, prediction)

            last_known_crime = prediction

            mask = (df['State'] == state) & (df['Year'] == year)
            if not df[mask].empty:
                df.loc[mask, target] = prediction
            else:
                new_entry = {k: new_row[k] for k in ['State', 'Year'] + features}
                new_entry[target] = prediction
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

    if 'Lag_Crime' in df.columns:
        df.drop(columns=['Lag_Crime'], inplace=True)

    return df.sort_values(['State', 'Year']).reset_index(drop=True)

def isolate_predicted_data(original, predicted):
    result = predicted.copy()

    # Ensure the DataFrames are aligned
    assert original.shape == predicted.shape
    assert (original[['State', 'Year']] == predicted[['State', 'Year']]).all().all(), \
        "State and Year must match between original and predicted data."

    # Columns to evaluate for predictions (exclude 'State' and 'Year')
    columns_to_check = [col for col in original.columns if col not in ['State', 'Year']]

    # Only keep values from predicted if they are NaN in original
    for col in columns_to_check:
        result[col] = predicted[col].where(original[col].isna(), np.nan)

    return result

def plot_state_group_features(STATES, selected_states=None):
    # Get all unique state names
    all_states = sorted(STATES['State'].unique())
    
    # If no specific states are given, show all
    if selected_states is None:
        selected_states = all_states

    if isinstance(selected_states, str):
        selected_states = [selected_states]

    # Filter the DataFrame to only include selected states
    group_df = STATES[STATES['State'].isin(selected_states)]

    if group_df.empty:
        print("No matching data found for the selected states.")
        return

    # Group by State and Year, averaging in case of duplicates
    grouped = group_df.groupby(['State', 'Year']).mean(numeric_only=True).reset_index()

    # Drop 'Year' and 'State' to get features to plot
    features = grouped.columns.drop(['State', 'Year'])

    # Setup subplots: one plot per feature
    n = len(features)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows), constrained_layout=True)
    axes = axes.flatten()

    # Plot each feature across the selected states
    for i, feature in enumerate(features):
        ax = axes[i]
        for state in selected_states:
            state_data = grouped[grouped['State'] == state]
            ax.plot(state_data['Year'], state_data[feature], label=state)

        ax.set_title(f"{feature}")
        ax.set_xlabel('Year')
        ax.set_ylabel(feature)
        ax.grid(True)

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a shared legend at the bottom
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', ncol=2, fontsize='medium', frameon=False)

    plt.subplots_adjust(bottom=2)  # Make space for legend

    plt.savefig('output.png')

def rank_states(df, year):
    data = df[df['Year'] == year][['State', 'Year', 'Population', 'CO2 Emissions (Millions of tonnes)', 'PPP', 'GDP', 'Average Salary', 'Average Disposable Income', 'Unemployment Rate', 'Traffic Fatalities', 'Air Quality Index', 'Violent Crimes Per Capita','Hospitals Per Capita']]
    data
    # Columns where a **higher value is better**
    higher_is_better = [
        'GDP',
        'PPP',
        'Average Salary',
        'Average Disposable Income',
        'Hospitals Per Capita'
    ]

    # Columns where a **lower value is better**
    lower_is_better = [
        'CO2 Emissions (Millions of tonnes)',
        'Unemployment Rate',
        'Traffic Fatalities',
        'Air Quality Index',
        'Violent Crimes Per Capita'
    ]

    ranked_df = data.copy()

    # Rank each column
    for col in higher_is_better:
        ranked_df[col + ' Rank'] = ranked_df[col].rank(method='min', ascending=False).astype(int)

    for col in lower_is_better:
        ranked_df[col + ' Rank'] = ranked_df[col].rank(method='min', ascending=True).astype(int)

    ranked_df = ranked_df[['State', 'Year', 'Population', 'CO2 Emissions (Millions of tonnes) Rank',
       'PPP Rank', 'GDP Rank', 'Average Salary Rank', 'Average Disposable Income Rank',
       'Hospitals Per Capita Rank',
       'Unemployment Rate Rank', 'Traffic Fatalities Rank',
       'Air Quality Index Rank', 'Violent Crimes Per Capita Rank']]
    rank_columns = [col for col in ranked_df.columns if 'Rank' in col]
    ranked_df['Final Score'] = ranked_df[rank_columns].sum(axis=1)
    return ranked_df.sort_values(by='Final Score')