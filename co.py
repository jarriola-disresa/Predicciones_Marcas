import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta, datetime
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import pymongo 
import hmac
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Dashboard de Predicción de Ventas Columbia",
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

#################################################################
# Password protection
# Password
def check_password():
    """Returns `True` if the user had the correct password."""
 
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False
 
    # Return True if the password is# validated.
    if st.session_state.get("password_correct", False):
        return True
 
    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False
 
 
if not check_password():
    st.stop()  # Do not continue if check_password is not True.


@st.cache_resource
def get_data():
    mongo_uri = st.secrets["mongouri"]
    client = pymongo.MongoClient(mongo_uri)
    db = client.Predicciones
    collection = db.COLUMBIA
    
    data = list(collection.find())
    df = pd.DataFrame(data)
    
    return df

def load_data():
    try:
        df = get_data()
        
        if df.empty:
            st.error("❌ No se encontraron datos en la colección MongoDB.")
            st.stop()
        
        df = df.drop(columns=['_id'], errors='ignore')
        
        if 'Fecha' not in df.columns:
            st.error("❌ No se encontró la columna 'Fecha' en los datos.")
            st.stop()
        
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Fecha'])
        
        df = df[df['Fecha'] >= '2023-01-01']
        
        numeric_cols = ['Cantidad']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        st.stop()

@st.cache_data
def get_summary_stats(df):
    return {
        'total_records': len(df),
        'date_range': (df['Fecha'].min(), df['Fecha'].max()),
        'total_quantity': df['Cantidad'].sum(),
        'avg_daily_sales': df.groupby('Fecha')['Cantidad'].sum().mean(),
        'unique_countries': df['Pais'].nunique(),
        'unique_stores': df['Bodega'].nunique()
    }

def create_kpi_cards(stats):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📦 Total Ventas",
            value=f"{stats['total_quantity']:,.0f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="📅 Promedio Diario",
            value=f"{stats['avg_daily_sales']:,.0f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="🌍 Países",
            value=f"{stats['unique_countries']}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="🏢 Bodegas",
            value=f"{stats['unique_stores']}",
            delta=None
        )

def apply_filters(df):
    with st.sidebar:
        st.header("🎯 Filtros de Datos")
        
        date_range = st.date_input(
            "📅 Rango de Fechas",
            value=(df['Fecha'].min(), df['Fecha'].max()),
            min_value=df['Fecha'].min(),
            max_value=df['Fecha'].max()
        )
        
        if len(date_range) == 2:
            df = df[(df['Fecha'] >= pd.to_datetime(date_range[0])) & 
                   (df['Fecha'] <= pd.to_datetime(date_range[1]))]
        
        pais = st.multiselect(
            "🌍 País",
            options=sorted(df['Pais'].dropna().unique()),
            default=None
        )
        
        bodega = st.multiselect(
            "🏢 Bodega",
            options=sorted(df['Bodega'].dropna().unique()),
            default=None
        )
        
        color = st.multiselect(
            "🎨 Color",
            options=sorted(df['U_Descrip_Color'].dropna().unique()),
            default=None
        )
        
        temporada = st.multiselect(
            "🗓️ Temporada",
            options=sorted(df['CL_Season'].dropna().unique()),
            default=None
        )
        
        with st.expander("🔧 Filtros Avanzados"):
            estilo = st.multiselect(
                "👗 Estilo",
                options=sorted(df['U_Estilo'].dropna().unique()),
                default=None
            )
            
            genero = st.multiselect(
                "👤 Género",
                options=sorted(df['U_Genero'].dropna().unique()),
                default=None
            )
            
            descripcion = st.multiselect(
                "📋 Descripción",
                options=sorted(df['U_Descripcion'].dropna().unique()),
                default=None
            )
            
            division = st.multiselect(
                "📂 División",
                options=sorted(df['U_Division'].dropna().unique()),
                default=None
            )
            
            categoria = st.multiselect(
                "📦 Categoría",
                options=sorted(df['U_Categoria'].dropna().unique()),
                default=None
            )
    
    if pais:
        df = df[df['Pais'].isin(pais)]
    if bodega:
        df = df[df['Bodega'].isin(bodega)]
    if color:
        df = df[df['U_Descrip_Color'].isin(color)]
    if temporada:
        df = df[df['CL_Season'].isin(temporada)]
    if estilo:
        df = df[df['U_Estilo'].isin(estilo)]
    if genero:
        df = df[df['U_Genero'].isin(genero)]
    if descripcion:
        df = df[df['U_Descripcion'].isin(descripcion)]
    if division:
        df = df[df['U_Division'].isin(division)]
    if categoria:
        df = df[df['U_Categoria'].isin(categoria)]
    
    return df

def create_historical_charts(df):
    st.subheader("📈 Análisis Histórico")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Tendencia Temporal", "Por País", "Por Temporada", "Distribución"])
    
    with tab1:
        daily_sales = df.groupby('Fecha')['Cantidad'].sum().reset_index()
        fig = px.line(daily_sales, x='Fecha', y='Cantidad', 
                     title='Ventas Diarias Históricas',
                     line_shape='spline')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        country_sales = df.groupby(['Fecha', 'Pais'])['Cantidad'].sum().reset_index()
        fig = px.line(country_sales, x='Fecha', y='Cantidad', color='Pais',
                     title='Ventas por País')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        season_sales = df.groupby(['Fecha', 'CL_Season'])['Cantidad'].sum().reset_index()
        fig = px.area(season_sales, x='Fecha', y='Cantidad', color='CL_Season',
                     title='Ventas por Temporada')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            top_products = df.groupby('U_Descripcion')['Cantidad'].sum().nlargest(10)
            fig = px.bar(x=top_products.values, y=top_products.index, 
                        orientation='h', title='Top 10 Productos')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            monthly_dist = df.groupby(df['Fecha'].dt.month)['Cantidad'].sum()
            month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                          'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            fig = px.bar(x=[month_names[i-1] for i in monthly_dist.index], 
                        y=monthly_dist.values,
                        title='Distribución Mensual')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def calculate_model_metrics(df, unidad_tiempo):
    """Calcula métricas de evaluación del modelo usando validación histórica"""
    metrics_results = []
    
    if unidad_tiempo == 'Diario':
        df['Periodo'] = df['Fecha']
        df_model = df.groupby(['Pais', 'Bodega', 'CL_Season', 'Periodo'])['Cantidad'].sum().reset_index()
        
        groups = list(df_model.groupby(['Pais', 'Bodega', 'CL_Season']))
        
        # Limitar a máximo 10 grupos para velocidad
        if len(groups) > 10:
            groups = groups[:10]
            st.info(f"📊 Evaluando una muestra de 10 segmentos de {len(list(df_model.groupby(['Pais', 'Bodega', 'CL_Season'])))} totales para mayor velocidad")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, ((pais_val, bodega_val, season_val), group) in enumerate(groups):
            status_text.text(f'Evaluando: {pais_val} - {bodega_val} - {season_val} ({idx+1}/{len(groups)})')
            progress_bar.progress((idx + 1) / len(groups))
            group = group.sort_values('Periodo')
            if group.shape[0] < 30:  # Necesitamos suficientes datos
                continue
            
            # Dividir en train/test (80/20)
            split_idx = int(len(group) * 0.8)
            train_data = group.iloc[:split_idx].copy()
            test_data = group.iloc[split_idx:].copy()
            
            if len(test_data) < 5:  # Mínimo de datos de prueba
                continue
            
            # Preparar características para entrenamiento
            train_data['Dia_Num'] = (train_data['Periodo'] - train_data['Periodo'].min()).dt.days
            train_data['weekday'] = train_data['Periodo'].dt.weekday
            train_data['is_weekend'] = train_data['weekday'].isin([5, 6]).astype(int)
            train_data['month'] = train_data['Periodo'].dt.month
            train_data['day'] = train_data['Periodo'].dt.day
            train_data['quarter'] = train_data['Periodo'].dt.quarter
            train_data['weekofyear'] = train_data['Periodo'].dt.isocalendar().week.astype(int)
            train_data['year'] = train_data['Periodo'].dt.year
            
            train_data['lag_7'] = train_data['Cantidad'].shift(7)
            train_data['lag_30'] = train_data['Cantidad'].shift(30)
            train_data['rolling_7'] = train_data['Cantidad'].rolling(window=7, min_periods=1).mean()
            train_data['rolling_30'] = train_data['Cantidad'].rolling(window=30, min_periods=1).mean()
            train_data = train_data.fillna(train_data['Cantidad'].mean())
            
            # Entrenar modelo (versión más rápida)
            X_train = train_data[['Dia_Num', 'weekday', 'is_weekend', 'month', 'day', 'quarter', 'weekofyear', 'year', 'lag_7', 'lag_30', 'rolling_7', 'rolling_30']]
            y_train = train_data['Cantidad']
            
            model = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
            model.fit(X_train, y_train)
            
            # Preparar datos de prueba
            test_predictions = []
            for _, row in test_data.iterrows():
                dia_num_pred = (row['Periodo'] - train_data['Periodo'].min()).days
                weekday_pred = row['Periodo'].weekday()
                is_weekend_pred = int(weekday_pred in [5, 6])
                month_pred = row['Periodo'].month
                day_pred = row['Periodo'].day
                quarter_pred = (row['Periodo'].month - 1) // 3 + 1
                weekofyear_pred = row['Periodo'].isocalendar()[1]
                year_pred = row['Periodo'].year
                
                last_values = train_data.tail(30)
                lag_7_val = last_values['Cantidad'].mean()
                lag_30_val = train_data['Cantidad'].mean()
                rolling_7_val = last_values['Cantidad'].mean()
                rolling_30_val = train_data['Cantidad'].mean()
                
                X_test = pd.DataFrame({
                    'Dia_Num': [dia_num_pred],
                    'weekday': [weekday_pred],
                    'is_weekend': [is_weekend_pred],
                    'month': [month_pred],
                    'day': [day_pred],
                    'quarter': [quarter_pred],
                    'weekofyear': [weekofyear_pred],
                    'year': [year_pred],
                    'lag_7': [lag_7_val],
                    'lag_30': [lag_30_val],
                    'rolling_7': [rolling_7_val],
                    'rolling_30': [rolling_30_val]
                })
                
                pred = model.predict(X_test)[0]
                test_predictions.append(max(pred, 0))
            
            # Calcular métricas
            y_true = test_data['Cantidad'].values
            y_pred = np.array(test_predictions)
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
            
            metrics_results.append({
                'Pais': pais_val,
                'Bodega': bodega_val,
                'CL_Season': season_val,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'Test_Size': len(test_data),
                'Actual_Mean': y_true.mean(),
                'Predicted_Mean': y_pred.mean()
            })
    
    elif unidad_tiempo == 'Mensual':
        df['Periodo'] = df['Fecha'].dt.to_period('M').dt.to_timestamp()
        df['Año'] = df['Periodo'].dt.year
        df['Mes'] = df['Periodo'].dt.month
        df['Mes_Num'] = df['Año'] * 12 + df['Mes']
        df['Mes_sin'] = np.sin(2 * np.pi * df['Mes'] / 12)
        df['Mes_cos'] = np.cos(2 * np.pi * df['Mes'] / 12)
        
        df_model = df.groupby(['Pais', 'Bodega', 'CL_Season', 'Periodo', 'Año', 'Mes', 'Mes_Num', 'Mes_sin', 'Mes_cos'])['Cantidad'].sum().reset_index()
        
        groups = list(df_model.groupby(['Pais', 'Bodega', 'CL_Season']))
        
        # Limitar a máximo 8 grupos para velocidad
        if len(groups) > 8:
            groups = groups[:8]
            st.info(f"📊 Evaluando una muestra de 8 segmentos de {len(list(df_model.groupby(['Pais', 'Bodega', 'CL_Season'])))} totales para mayor velocidad")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, ((pais_val, bodega_val, season_val), group) in enumerate(groups):
            status_text.text(f'Evaluando: {pais_val} - {bodega_val} - {season_val} ({idx+1}/{len(groups)})')
            progress_bar.progress((idx + 1) / len(groups))
            group = group.sort_values('Periodo')
            if group.shape[0] < 12:  # Necesitamos al menos 12 meses
                continue
            
            # Dividir en train/test (80/20)
            split_idx = int(len(group) * 0.8)
            train_data = group.iloc[:split_idx].copy()
            test_data = group.iloc[split_idx:].copy()
            
            if len(test_data) < 3:  # Mínimo 3 meses de prueba
                continue
            
            # Entrenar modelo (versión más rápida)
            X_train = train_data[['Mes_Num', 'Año', 'Mes_sin', 'Mes_cos']]
            y_train = train_data['Cantidad']
            
            model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.15, random_state=42, n_jobs=-1, verbosity=0)
            model.fit(X_train, y_train)
            
            # Predecir en datos de prueba
            X_test = test_data[['Mes_Num', 'Año', 'Mes_sin', 'Mes_cos']]
            y_pred = model.predict(X_test)
            y_pred = np.maximum(y_pred, 0)  # No negativos
            
            # Calcular métricas
            y_true = test_data['Cantidad'].values
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
            
            metrics_results.append({
                'Pais': pais_val,
                'Bodega': bodega_val,
                'CL_Season': season_val,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'Test_Size': len(test_data),
                'Actual_Mean': y_true.mean(),
                'Predicted_Mean': y_pred.mean()
            })
        
        progress_bar.empty()
        status_text.empty()
    
    return pd.DataFrame(metrics_results)

def predict_sales_original(df, unidad_tiempo, fecha_inicio_pred, fecha_fin_pred):
    predicciones = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if unidad_tiempo == 'Diario':
        peso_promedio = 0.3
        peso_modelo = 0.7

        df['weekday'] = df['Fecha'].dt.weekday
        fines_df = df[df['weekday'].isin([5, 6])].copy()
        fines_df['year'] = fines_df['Fecha'].dt.year
        fines_df = fines_df[fines_df['year'].isin([2023, 2024, 2025])]
        promedios_fines = (
            fines_df.groupby(['Pais', 'Bodega', 'CL_Season', 'weekday'])['Cantidad']
            .mean()
            .reset_index()
            .rename(columns={'Cantidad': 'Promedio_Finde'})
        )

        df['Periodo'] = df['Fecha']
        df_model = df.groupby(['Pais', 'Bodega', 'CL_Season', 'Periodo'])['Cantidad'].sum().reset_index()

        groups = list(df_model.groupby(['Pais', 'Bodega', 'CL_Season']))
        total_groups = len(groups)
        
        for idx, ((pais_val, bodega_val, season_val), group) in enumerate(groups):
            status_text.text(f'Procesando: {pais_val} - {bodega_val} - {season_val} ({idx+1}/{total_groups})')
            progress_bar.progress((idx + 1) / total_groups)
            
            group = group.sort_values('Periodo')
            if group.shape[0] < 2:
                continue

            group['Dia_Num'] = (group['Periodo'] - group['Periodo'].min()).dt.days
            group['weekday'] = group['Periodo'].dt.weekday
            group['is_weekend'] = group['weekday'].isin([5, 6]).astype(int)
            group['month'] = group['Periodo'].dt.month
            group['day'] = group['Periodo'].dt.day
            group['quarter'] = group['Periodo'].dt.quarter
            group['weekofyear'] = group['Periodo'].dt.isocalendar().week.astype(int)
            group['year'] = group['Periodo'].dt.year
            
            group['lag_7'] = group['Cantidad'].shift(7)
            group['lag_30'] = group['Cantidad'].shift(30)
            group['rolling_7'] = group['Cantidad'].rolling(window=7, min_periods=1).mean()
            group['rolling_30'] = group['Cantidad'].rolling(window=30, min_periods=1).mean()
            
            recent_trend = 1.08 if len(group) > 30 and group['Cantidad'].tail(30).mean() > group['Cantidad'].head(30).mean() else 1.02
            
            year_trend = 1.0
            if fecha_inicio_pred.year >= 2025:
                if len(group) > 365:
                    data_2024 = group[group['Periodo'].dt.year == 2024]['Cantidad']
                    data_2023 = group[group['Periodo'].dt.year == 2023]['Cantidad']
                    if len(data_2024) > 0 and len(data_2023) > 0:
                        growth_rate = data_2024.mean() / data_2023.mean() if data_2023.mean() > 0 else 1.0
                        year_trend = max(growth_rate, 1.05)
            
            recent_trend = recent_trend * year_trend
            
            group = group.fillna(group['Cantidad'].mean())

            X_train = group[['Dia_Num', 'weekday', 'is_weekend', 'month', 'day', 'quarter', 'weekofyear', 'year', 'lag_7', 'lag_30', 'rolling_7', 'rolling_30']]
            y_train = group['Cantidad']

            model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1, verbosity=0)
            model.fit(X_train, y_train)

            for i in range((fecha_fin_pred - fecha_inicio_pred).days + 1):
                fecha = fecha_inicio_pred + timedelta(days=i)
                dia_num_pred = (fecha - group['Periodo'].min()).days
                weekday_pred = fecha.weekday()
                is_weekend_pred = int(weekday_pred in [5, 6])
                month_pred = fecha.month
                day_pred = fecha.day
                quarter_pred = (fecha.month - 1) // 3 + 1
                weekofyear_pred = fecha.isocalendar()[1]
                year_pred = fecha.year
                
                last_values = group.tail(30)
                lag_7_val = last_values['Cantidad'].mean()
                lag_30_val = group['Cantidad'].mean()
                rolling_7_val = last_values['Cantidad'].mean()
                rolling_30_val = group['Cantidad'].mean()

                X_pred = pd.DataFrame({
                    'Dia_Num': [dia_num_pred],
                    'weekday': [weekday_pred],
                    'is_weekend': [is_weekend_pred],
                    'month': [month_pred],
                    'day': [day_pred],
                    'quarter': [quarter_pred],
                    'weekofyear': [weekofyear_pred],
                    'year': [year_pred],
                    'lag_7': [lag_7_val],
                    'lag_30': [lag_30_val],
                    'rolling_7': [rolling_7_val],
                    'rolling_30': [rolling_30_val]
                })

                pred_model = model.predict(X_pred)[0] * recent_trend

                if is_weekend_pred:
                    match = promedios_fines[
                        (promedios_fines['Pais'] == pais_val) &
                        (promedios_fines['Bodega'] == bodega_val) &
                        (promedios_fines['CL_Season'] == season_val) &
                        (promedios_fines['weekday'] == weekday_pred)
                    ]
                    if not match.empty:
                        prom = match['Promedio_Finde'].values[0]
                        pred = max(peso_promedio * prom + peso_modelo * pred_model, 0)
                    else:
                        pred = max(pred_model, 0)
                else:
                    pred = max(pred_model, 0)

                predicciones.append({
                    'Pais': pais_val,
                    'Bodega': bodega_val,
                    'CL_Season': season_val,
                    'Periodo': fecha,
                    'Predicción': pred
                })

    elif unidad_tiempo == 'Mensual':
        df['Periodo'] = df['Fecha'].dt.to_period('M').dt.to_timestamp()
        df['Año'] = df['Periodo'].dt.year
        df['Mes'] = df['Periodo'].dt.month
        df['Mes_Num'] = df['Año'] * 12 + df['Mes']
        df['Mes_sin'] = np.sin(2 * np.pi * df['Mes'] / 12)
        df['Mes_cos'] = np.cos(2 * np.pi * df['Mes'] / 12)

        df_model = df.groupby(['Pais', 'Bodega', 'CL_Season', 'Periodo', 'Año', 'Mes', 'Mes_Num', 'Mes_sin', 'Mes_cos'])['Cantidad'].sum().reset_index()

        groups = list(df_model.groupby(['Pais', 'Bodega', 'CL_Season']))
        total_groups = len(groups)
        
        for idx, ((pais_val, bodega_val, season_val), group) in enumerate(groups):
            status_text.text(f'Procesando: {pais_val} - {bodega_val} - {season_val} ({idx+1}/{total_groups})')
            progress_bar.progress((idx + 1) / total_groups)
            
            group = group.sort_values('Periodo')
            if group.shape[0] < 2:
                continue

            recent_trend = 1.05 if len(group) > 12 and group['Cantidad'].tail(6).mean() > group['Cantidad'].head(6).mean() else 1.02
            
            year_trend = 1.0
            if fecha_inicio_pred.year >= 2025:
                if len(group) > 12:
                    data_2024 = group[group['Periodo'].dt.year == 2024]['Cantidad']
                    data_2023 = group[group['Periodo'].dt.year == 2023]['Cantidad']
                    if len(data_2024) > 0 and len(data_2023) > 0:
                        growth_rate = data_2024.mean() / data_2023.mean() if data_2023.mean() > 0 else 1.0
                        year_trend = max(growth_rate, 1.03)
            
            recent_trend = recent_trend * year_trend

            X_train = group[['Mes_Num', 'Año', 'Mes_sin', 'Mes_cos']]
            y_train = group['Cantidad']

            model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
            model.fit(X_train, y_train)

            for anio_pred in range(fecha_inicio_pred.year, fecha_fin_pred.year + 1):
                start_mes = fecha_inicio_pred.month if anio_pred == fecha_inicio_pred.year else 1
                end_mes = fecha_fin_pred.month if anio_pred == fecha_fin_pred.year else 12
                for mes in range(start_mes, end_mes + 1):
                    mes_num_pred = anio_pred * 12 + mes
                    X_pred = pd.DataFrame({
                        'Mes_Num': [mes_num_pred],
                        'Año': [anio_pred],
                        'Mes_sin': [np.sin(2 * np.pi * mes / 12)],
                        'Mes_cos': [np.cos(2 * np.pi * mes / 12)]
                    })
                    pred = model.predict(X_pred)[0] * recent_trend
                    pred = max(pred, 0)
                    predicciones.append({
                        'Pais': pais_val,
                        'Bodega': bodega_val,
                        'CL_Season': season_val,
                        'Periodo': pd.Timestamp(year=anio_pred, month=mes, day=1),
                        'Predicción': pred
                    })
    
    progress_bar.empty()
    status_text.empty()
    
    return predicciones

def display_predictions(predicciones, df_model, unidad_tiempo):
    if not predicciones:
        st.warning("⚠️ No se generaron predicciones con los filtros actuales.")
        return
    
    resultados = pd.DataFrame(predicciones)
    
    st.subheader("🔮 Resultados de Predicción")
    
    tab1, tab2, tab3 = st.tabs(["Gráfico Principal", "Análisis Detallado", "Datos Exportables"])
    
    with tab1:
        historico = df_model.groupby('Periodo')['Cantidad'].sum().reset_index()
        pred_total = resultados.groupby('Periodo')['Predicción'].sum().reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historico['Periodo'], 
            y=historico['Cantidad'],
            mode='lines',
            name='Histórico'
        ))
        
        fig.add_trace(go.Scatter(
            x=pred_total['Periodo'], 
            y=pred_total['Predicción'],
            mode='lines',
            name='Predicción'
        ))
        
        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Cantidad",
            legend=dict(x=0, y=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            pred_by_country = resultados.groupby('Pais')['Predicción'].sum().sort_values(ascending=False)
            fig = px.pie(values=pred_by_country.values, names=pred_by_country.index,
                        title='Predicción por País')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            pred_by_season = resultados.groupby('CL_Season')['Predicción'].sum().sort_values(ascending=False)
            fig = px.bar(x=pred_by_season.index, y=pred_by_season.values,
                        title='Predicción por Temporada')
            st.plotly_chart(fig, use_container_width=True)
        
        monthly_pred = resultados.copy()
        monthly_pred['Mes'] = monthly_pred['Periodo'].dt.month
        monthly_summary = monthly_pred.groupby('Mes')['Predicción'].sum()
        
        month_names = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                      7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
        
        fig = px.line(
            x=[month_names[m] for m in monthly_summary.index],
            y=monthly_summary.values,
            title='Patrón de Predicción Mensual',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparación estadística
        if len(pred_total) > 0:
            historico_stats = historico['Cantidad'].describe()
            pred_stats = pred_total['Predicción'].describe()
            
            comparison_df = pd.DataFrame({
                'Histórico': historico_stats,
                'Predicción': pred_stats
            })
            
            st.write("📊 Comparación Estadística:")
            st.dataframe(comparison_df.round(2))
            
            st.write("**📋 Explicación de Indicadores:**")
            st.write("• **count**: Número total de períodos analizados")
            st.write("• **mean**: Promedio de ventas por período")
            st.write("• **std**: Desviación estándar (variabilidad de las ventas)")
            st.write("• **min**: Valor mínimo registrado")
            st.write("• **25%**: Primer cuartil (25% de los valores están por debajo)")
            st.write("• **50%**: Mediana (valor que divide los datos por la mitad)")
            st.write("• **75%**: Tercer cuartil (75% de los valores están por debajo)")
            st.write("• **max**: Valor máximo registrado")
    
    with tab3:
        display_df = resultados.copy()
        display_df['Predicción'] = display_df['Predicción'].round().astype(int)
        
        if unidad_tiempo == 'Mensual':
            display_df['Periodo'] = display_df['Periodo'].dt.strftime('%B %Y')
        else:
            display_df['Periodo'] = display_df['Periodo'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_df, use_container_width=True)
        
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Predicciones (CSV)",
            data=csv,
            file_name=f'predicciones_{unidad_tiempo.lower()}_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )

def main():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("📊 Dashboard de Predicción de Ventas Columbia")
    
    with col2:
        st.markdown("""
        <div style="text-align: right; padding: 10px;">
            <div style="font-size: 24px; font-weight: bold; color: #1f4e79; margin-bottom: 5px;">
                🏔️ COLUMBIA
            </div>
            <div style="font-size: 12px; color: #666; font-style: italic;">
                Sportswear Company
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    df = load_data()
    stats = get_summary_stats(df)
    
    st.subheader("📋 Resumen Ejecutivo")
    create_kpi_cards(stats)
    st.markdown("---")
    
    df_filtered = apply_filters(df)
    
    if df_filtered.empty:
        st.warning("⚠️ No hay datos para los filtros seleccionados. Modifica los filtros en la barra lateral.")
        return
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔮 Configuración de Predicción")
    
    unidad_tiempo = st.sidebar.selectbox(
        "⏳ Unidad de Tiempo",
        ['Diario', 'Mensual'],
        help="Selecciona la granularidad temporal para las predicciones"
    )
    
    fecha_inicio_pred = st.sidebar.date_input(
        '📅 Inicio Predicción',
        pd.Timestamp('2025-06-26'),
        help="Fecha de inicio para las predicciones"
    )
    
    fecha_fin_pred = st.sidebar.date_input(
        '📅 Fin Predicción',
        pd.Timestamp('2025-12-31'),
        help="Fecha final para las predicciones"
    )
    
    create_historical_charts(df_filtered)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Generar Predicciones", type="primary"):
            with st.spinner('🤖 Generando predicciones...'):
                predicciones = predict_sales_original(
                    df_filtered.copy(), 
                    unidad_tiempo, 
                    pd.to_datetime(fecha_inicio_pred), 
                    pd.to_datetime(fecha_fin_pred)
                )
                
                if unidad_tiempo == 'Diario':
                    df_model = df_filtered.groupby(['Pais', 'Bodega', 'CL_Season', 'Fecha'])['Cantidad'].sum().reset_index()
                    df_model.rename(columns={'Fecha': 'Periodo'}, inplace=True)
                else:
                    df_filtered['Periodo'] = df_filtered['Fecha'].dt.to_period('M').dt.to_timestamp()
                    df_model = df_filtered.groupby(['Pais', 'Bodega', 'CL_Season', 'Periodo'])['Cantidad'].sum().reset_index()
                
                # Guardar en session_state
                st.session_state['predicciones'] = predicciones
                st.session_state['df_model'] = df_model
                st.session_state['unidad_tiempo'] = unidad_tiempo
    
    with col2:
        if st.button("📊 Evaluar Precisión del Modelo", type="secondary"):
            with st.spinner('🔍 Calculando métricas de evaluación...'):
                metrics_df = calculate_model_metrics(df_filtered.copy(), unidad_tiempo)
                
                # Guardar métricas en session_state
                st.session_state['metrics_df'] = metrics_df
                
                if not metrics_df.empty:
                    st.success("✅ Evaluación completada")
                
                else:
                    st.warning("⚠️ No se pudieron calcular métricas. Verifica que haya suficientes datos históricos.")
    
    # Mostrar predicciones si están disponibles
    if 'predicciones' in st.session_state and st.session_state['predicciones']:
        display_predictions(st.session_state['predicciones'], st.session_state['df_model'], st.session_state['unidad_tiempo'])
    
    # Mostrar métricas si están disponibles
    if 'metrics_df' in st.session_state and not st.session_state['metrics_df'].empty:
        metrics_df = st.session_state['metrics_df']
        
        st.subheader("📈 Métricas de Evaluación del Modelo")
        
        # Métricas promedio
        avg_mae = metrics_df['MAE'].mean()
        avg_rmse = metrics_df['RMSE'].mean()
        avg_mape = metrics_df['MAPE'].mean()
        total_tests = metrics_df['Test_Size'].sum()
        accuracy_pct = 100 - avg_mape
        
        col1_m, col2_m, col3_m, col4_m, col5_m = st.columns(5)
        
        with col1_m:
            st.metric("MAE Promedio", f"{avg_mae:.1f}", help="Error Absoluto Medio")
        with col2_m:
            st.metric("RMSE Promedio", f"{avg_rmse:.1f}", help="Raíz del Error Cuadrático Medio")
        with col3_m:
            st.metric("MAPE Promedio", f"{avg_mape:.1f}%", help="Error Porcentual Absoluto Medio")
        with col4_m:
            st.metric("Precisión", f"{accuracy_pct:.1f}%", help="Precisión general del modelo")
        with col5_m:
            st.metric("Datos Evaluados", f"{total_tests}", help="Total de observaciones evaluadas")
        
        st.markdown("---")
        st.write("### 📋 ¿Qué significan estas métricas?")
        
        col1_exp, col2_exp, col3_exp = st.columns(3)
        
        with col1_exp:
            st.write("**🎯 MAE (Error Absoluto Medio)**")
            st.write(f"• En promedio, el modelo se equivoca por **{avg_mae:.1f} unidades**")
            st.write("• Ejemplo: Si predice 100, el valor real podría ser entre 90-110")
            
        with col2_exp:
            st.write("**📊 RMSE (Error Cuadrático Medio)**")
            st.write(f"• Penaliza más los errores grandes: **{avg_rmse:.1f}**")
            st.write("• Si RMSE ≈ MAE → errores consistentes")
            st.write("• Si RMSE >> MAE → algunos errores muy grandes")
            
        with col3_exp:
            st.write("**📈 MAPE (Error Porcentual)**")
            st.write(f"• El modelo tiene **{avg_mape:.1f}% de error** en promedio")
            st.write(f"• **Precisión del {accuracy_pct:.1f}%**")
            if avg_mape < 10:
                st.success("🎯 Excelente precisión")
            elif avg_mape < 20:
                st.info("✔️ Buena precisión")
            elif avg_mape < 30:
                st.warning("⚠️ Precisión regular")
            else:
                st.error("❌ Baja precisión")


if __name__ == "__main__":
    main()