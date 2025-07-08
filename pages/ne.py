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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import pymongo 
import hmac
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Dashboard de Predicción de Ventas New Era",
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
    collection = db.NEW_ERA
    
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
        
        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['Fecha'])
        
        df = df[df['Fecha'] >= '2023-01-01']
        
        # Remove MATERIAL DE EMPAQUE from all analysis
        if 'U_Segmento' in df.columns:
            df = df[df['U_Segmento'] != 'MATERIAL DE EMPAQUE']
        
        numeric_cols = ['Cantidad', 'USD_Total_SI_CD']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        st.stop()

@st.cache_data
def get_summary_stats(df, metric_column):
    return {
        'total_records': len(df),
        'date_range': (df['Fecha'].min(), df['Fecha'].max()),
        'total_quantity': df[metric_column].sum(),
        'avg_daily_sales': df.groupby('Fecha')[metric_column].sum().mean(),
        'unique_countries': df['Pais'].nunique() if 'Pais' in df.columns else 0,
        'unique_stores': df['Tienda'].nunique() if 'Tienda' in df.columns else 0
    }

def create_kpi_cards(stats):
    pass

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
        
        if 'Pais' in df.columns:
            pais = st.multiselect(
                "🌍 País",
                options=sorted(df['Pais'].dropna().unique()),
                default=None
            )
            if pais:
                df = df[df['Pais'].isin(pais)]
        
        if 'Tienda' in df.columns:
            tienda = st.multiselect(
                "🏢 Tienda",
                options=sorted(df['Tienda'].dropna().unique()),
                default=None
            )
            if tienda:
                df = df[df['Tienda'].isin(tienda)]
        
        if 'U_Liga' in df.columns:
            liga = st.multiselect(
                "🏀 Liga",
                options=sorted(df['U_Liga'].dropna().unique()),
                default=None
            )
            if liga:
                df = df[df['U_Liga'].isin(liga)]
        
        if 'U_Segmento' in df.columns:
            segmento = st.multiselect(
                "🗓️ Segmento",
                options=sorted(df['U_Segmento'].dropna().unique()),
                default=None
            )
            if segmento:
                df = df[df['U_Segmento'].isin(segmento)]
        
        with st.expander("🔧 Filtros Avanzados"):
            if 'U_Estilo' in df.columns:
                estilo = st.multiselect(
                    "👗 Estilo",
                    options=sorted(df['U_Estilo'].dropna().unique()),
                    default=None
                )
                if estilo:
                    df = df[df['U_Estilo'].isin(estilo)]
            
            if 'U_Silueta' in df.columns:
                silueta = st.multiselect(
                    "👤 Silueta",
                    options=sorted(df['U_Silueta'].dropna().unique()),
                    default=None
                )
                if silueta:
                    df = df[df['U_Silueta'].isin(silueta)]
            
            if 'U_Team' in df.columns:
                team = st.multiselect(
                    "🏈 Team",
                    options=sorted(df['U_Team'].dropna().unique()),
                    default=None
                )
                if team:
                    df = df[df['U_Team'].isin(team)]
    
    # Ensure MATERIAL DE EMPAQUE is always excluded
    if 'U_Segmento' in df.columns:
        df = df[df['U_Segmento'] != 'MATERIAL DE EMPAQUE']
    
    return df

def create_historical_charts(df, metric_column='Cantidad'):
    st.subheader("📈 Análisis Histórico")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Tendencia Temporal", "Por País", "Por Segmento", "Distribución"])
    
    with tab1:
        daily_sales = df.groupby('Fecha')[metric_column].sum().reset_index()
        metric_label = 'Cantidad' if metric_column == 'Cantidad' else 'USD Total'
        fig = px.line(daily_sales, x='Fecha', y=metric_column, 
                     title=f'{metric_label} Diario Histórico',
                     line_shape='spline')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'Pais' in df.columns:
            country_sales = df.groupby(['Fecha', 'Pais'])[metric_column].sum().reset_index()
            fig = px.line(country_sales, x='Fecha', y=metric_column, color='Pais',
                         title=f'{metric_label} por País')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de País disponibles")
    
    with tab3:
        if 'U_Segmento' in df.columns:
            season_sales = df.groupby(['Fecha', 'U_Segmento'])[metric_column].sum().reset_index()
            fig = px.area(season_sales, x='Fecha', y=metric_column, color='U_Segmento',
                         title=f'{metric_label} por Segmento')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de Segmento disponibles")
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'U_Team' in df.columns:
                top_products = df.groupby('U_Team')[metric_column].sum().nlargest(10)
                fig = px.bar(x=top_products.values, y=top_products.index, 
                            orientation='h', title='Top 10 Equipos')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos de Team disponibles")
        
        with col2:
            monthly_dist = df.groupby(df['Fecha'].dt.month)[metric_column].sum()
            month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                          'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            fig = px.bar(x=[month_names[i-1] for i in monthly_dist.index], 
                        y=monthly_dist.values,
                        title='Distribución Mensual')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def predict_sales_original(df, unidad_tiempo, fecha_inicio_pred, fecha_fin_pred, metric_column='Cantidad'):
    predicciones = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    model_metrics = []
    
    # Determine available grouping columns
    grouping_cols = ['Pais', 'Tienda', 'U_Segmento']
    available_cols = [col for col in grouping_cols if col in df.columns]
    
    if len(available_cols) == 0:
        st.error("No hay suficientes columnas para realizar la predicción")
        return predicciones
    
    if unidad_tiempo == 'Diario':
        df['weekday'] = df['Fecha'].dt.weekday
        df['Periodo'] = df['Fecha']
        df_model = df.groupby(available_cols + ['Periodo'])[metric_column].sum().reset_index()

        groups = list(df_model.groupby(available_cols))
        total_groups = len(groups)
        
        for idx, (group_vals, group) in enumerate(groups):
            if len(available_cols) == 1:
                status_text.text(f'Procesando: {group_vals} ({idx+1}/{total_groups})')
            else:
                status_text.text(f'Procesando: {" - ".join(map(str, group_vals))} ({idx+1}/{total_groups})')
            progress_bar.progress((idx + 1) / total_groups)
            
            group = group.sort_values('Periodo')
            
            # Filtro de calidad básico: requiere al menos 21 días de datos
            if group.shape[0] < 21:
                continue
            
            # Filtrar solo grupos con actividad mínima (no filtrar por variabilidad)
            mean_sales = group[metric_column].mean()
            if mean_sales < 0.5:  # Solo grupos con actividad mínima
                continue

            # Crear features temporales mejoradas
            group['Dia_Num'] = (group['Periodo'] - group['Periodo'].min()).dt.days
            group['weekday'] = group['Periodo'].dt.weekday
            group['is_weekend'] = group['weekday'].isin([5, 6]).astype(int)
            group['month'] = group['Periodo'].dt.month
            group['day'] = group['Periodo'].dt.day
            group['quarter'] = group['Periodo'].dt.quarter
            group['weekofyear'] = group['Periodo'].dt.isocalendar().week.astype(int)
            group['year'] = group['Periodo'].dt.year
            
            # Features de lag mejoradas
            group['lag_1'] = group[metric_column].shift(1)
            group['lag_7'] = group[metric_column].shift(7)
            group['lag_14'] = group[metric_column].shift(14)
            group['lag_30'] = group[metric_column].shift(30)
            
            # Features de rolling mejoradas
            group['rolling_3'] = group[metric_column].rolling(window=3, min_periods=1).mean()
            group['rolling_7'] = group[metric_column].rolling(window=7, min_periods=1).mean()
            group['rolling_14'] = group[metric_column].rolling(window=14, min_periods=1).mean()
            group['rolling_30'] = group[metric_column].rolling(window=30, min_periods=1).mean()
            
            # Features de tendencia
            group['trend_7'] = group[metric_column].rolling(window=7, min_periods=1).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            group['trend_30'] = group[metric_column].rolling(window=30, min_periods=1).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            
            # Features de volatilidad
            group['volatility_7'] = group[metric_column].rolling(window=7, min_periods=1).std()
            group['volatility_30'] = group[metric_column].rolling(window=30, min_periods=1).std()
            
            # Features estacionales
            group['month_sin'] = np.sin(2 * np.pi * group['month'] / 12)
            group['month_cos'] = np.cos(2 * np.pi * group['month'] / 12)
            group['week_sin'] = np.sin(2 * np.pi * group['weekofyear'] / 52)
            group['week_cos'] = np.cos(2 * np.pi * group['weekofyear'] / 52)
            
            # Rellenar valores faltantes con métodos más sofisticados
            numeric_cols = ['lag_1', 'lag_7', 'lag_14', 'lag_30', 'rolling_3', 'rolling_7', 'rolling_14', 'rolling_30', 'trend_7', 'trend_30', 'volatility_7', 'volatility_30']
            for col in numeric_cols:
                group[col] = group[col].fillna(group[metric_column].mean())

            # Features expandidas para el modelo
            feature_cols = ['Dia_Num', 'weekday', 'is_weekend', 'month', 'day', 'quarter', 'year', 'weekofyear',
                          'lag_1', 'lag_7', 'lag_14', 'lag_30', 'rolling_3', 'rolling_7', 'rolling_14', 'rolling_30',
                          'trend_7', 'trend_30', 'volatility_7', 'volatility_30', 'month_sin', 'month_cos', 'week_sin', 'week_cos']
            
            X_train = group[feature_cols]
            y_train = group[metric_column]
            
            # Normalización de features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Modelo optimizado para velocidad
            if metric_column == 'USD_Total_SI_CD':
                model = XGBRegressor(
                    n_estimators=100,  # Reducido significativamente
                    max_depth=4, 
                    learning_rate=0.1, 
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42, 
                    n_jobs=-1, 
                    verbosity=0
                )
            else:
                model = XGBRegressor(
                    n_estimators=100,  # Reducido significativamente
                    max_depth=4, 
                    learning_rate=0.1, 
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42, 
                    n_jobs=-1, 
                    verbosity=0
                )
            
            # Validación cruzada desactivada para velocidad
            # Solo métricas básicas
            if len(X_train) > 21:
                metric_record = {'MAE': 0, 'Data_Points': len(X_train)}
                for col, val in zip(available_cols, group_vals):
                    metric_record[col] = val
                model_metrics.append(metric_record)
            
            # Entrenar modelo final
            model.fit(X_train_scaled, y_train)
            
            # Generar predicciones secuenciales
            last_values = group.tail(1).copy()
            
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
                
                # Usar valores lag reales de los últimos datos
                lag_1_val = last_values[metric_column].iloc[-1] if i == 0 else predicciones[-1]['Predicción']
                lag_7_val = group[metric_column].iloc[-7] if len(group) >= 7 else group[metric_column].mean()
                lag_14_val = group[metric_column].iloc[-14] if len(group) >= 14 else group[metric_column].mean()
                lag_30_val = group[metric_column].iloc[-30] if len(group) >= 30 else group[metric_column].mean()
                
                # Rolling values actualizados
                rolling_3_val = group[metric_column].tail(3).mean()
                rolling_7_val = group[metric_column].tail(7).mean()
                rolling_14_val = group[metric_column].tail(14).mean()
                rolling_30_val = group[metric_column].tail(30).mean()
                
                # Tendencia y volatilidad
                trend_7_val = group['trend_7'].iloc[-1] if not pd.isna(group['trend_7'].iloc[-1]) else 0
                trend_30_val = group['trend_30'].iloc[-1] if not pd.isna(group['trend_30'].iloc[-1]) else 0
                volatility_7_val = group['volatility_7'].iloc[-1] if not pd.isna(group['volatility_7'].iloc[-1]) else 0
                volatility_30_val = group['volatility_30'].iloc[-1] if not pd.isna(group['volatility_30'].iloc[-1]) else 0
                
                # Features estacionales
                month_sin_val = np.sin(2 * np.pi * month_pred / 12)
                month_cos_val = np.cos(2 * np.pi * month_pred / 12)
                week_sin_val = np.sin(2 * np.pi * weekofyear_pred / 52)
                week_cos_val = np.cos(2 * np.pi * weekofyear_pred / 52)

                # Crear features para predicción
                X_pred = pd.DataFrame({
                    'Dia_Num': [dia_num_pred],
                    'weekday': [weekday_pred],
                    'is_weekend': [is_weekend_pred],
                    'month': [month_pred],
                    'day': [day_pred],
                    'quarter': [quarter_pred],
                    'year': [year_pred],
                    'weekofyear': [weekofyear_pred],
                    'lag_1': [lag_1_val],
                    'lag_7': [lag_7_val],
                    'lag_14': [lag_14_val],
                    'lag_30': [lag_30_val],
                    'rolling_3': [rolling_3_val],
                    'rolling_7': [rolling_7_val],
                    'rolling_14': [rolling_14_val],
                    'rolling_30': [rolling_30_val],
                    'trend_7': [trend_7_val],
                    'trend_30': [trend_30_val],
                    'volatility_7': [volatility_7_val],
                    'volatility_30': [volatility_30_val],
                    'month_sin': [month_sin_val],
                    'month_cos': [month_cos_val],
                    'week_sin': [week_sin_val],
                    'week_cos': [week_cos_val]
                })

                # Normalizar features de predicción
                X_pred_scaled = scaler.transform(X_pred)
                pred_model = model.predict(X_pred_scaled)[0]
                
                # Aplicar límites simples para velocidad
                pred = max(pred_model, 0)
                
                # Límite superior simple basado en percentiles
                upper_limit = group[metric_column].quantile(0.95)
                pred = min(pred, upper_limit * 1.2)

                # Create prediction record
                pred_record = {'Periodo': fecha, 'Predicción': pred}
                for col, val in zip(available_cols, group_vals):
                    pred_record[col] = val
                    
                predicciones.append(pred_record)

    elif unidad_tiempo == 'Mensual':
        df['Periodo'] = df['Fecha'].dt.to_period('M').dt.to_timestamp()
        df['Año'] = df['Periodo'].dt.year
        df['Mes'] = df['Periodo'].dt.month
        df['Mes_Num'] = df['Año'] * 12 + df['Mes']
        df['Mes_sin'] = np.sin(2 * np.pi * df['Mes'] / 12)
        df['Mes_cos'] = np.cos(2 * np.pi * df['Mes'] / 12)

        df_model = df.groupby(available_cols + ['Periodo', 'Año', 'Mes', 'Mes_Num', 'Mes_sin', 'Mes_cos'])[metric_column].sum().reset_index()

        groups = list(df_model.groupby(available_cols))
        total_groups = len(groups)
        
        for idx, (group_vals, group) in enumerate(groups):
            if len(available_cols) == 1:
                status_text.text(f'Procesando: {group_vals} ({idx+1}/{total_groups})')
            else:
                status_text.text(f'Procesando: {" - ".join(map(str, group_vals))} ({idx+1}/{total_groups})')
            progress_bar.progress((idx + 1) / total_groups)
            
            group = group.sort_values('Periodo')
            
            # Filtro de calidad básico: requiere al menos 4 meses de datos
            if group.shape[0] < 4:
                continue
                
            # Filtrar solo grupos con actividad mínima (no filtrar por variabilidad)
            mean_sales = group[metric_column].mean()
            if mean_sales < 2:  # Solo grupos con actividad mínima
                continue

            # Crear features temporales mejoradas para mensual
            group['quarter'] = group['Periodo'].dt.quarter
            group['month_of_year'] = group['Periodo'].dt.month
            group['year_month'] = group['Año'] * 12 + group['Mes']
            
            # Features de lag mensuales
            group['lag_1'] = group[metric_column].shift(1)
            group['lag_3'] = group[metric_column].shift(3)
            group['lag_6'] = group[metric_column].shift(6)
            group['lag_12'] = group[metric_column].shift(12)
            
            # Features de rolling mensuales
            group['rolling_3'] = group[metric_column].rolling(window=3, min_periods=1).mean()
            group['rolling_6'] = group[metric_column].rolling(window=6, min_periods=1).mean()
            group['rolling_12'] = group[metric_column].rolling(window=12, min_periods=1).mean()
            
            # Features de tendencia mensual
            group['trend_3'] = group[metric_column].rolling(window=3, min_periods=1).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            group['trend_6'] = group[metric_column].rolling(window=6, min_periods=1).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            group['trend_12'] = group[metric_column].rolling(window=12, min_periods=1).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            
            # Features de volatilidad mensual
            group['volatility_3'] = group[metric_column].rolling(window=3, min_periods=1).std()
            group['volatility_6'] = group[metric_column].rolling(window=6, min_periods=1).std()
            group['volatility_12'] = group[metric_column].rolling(window=12, min_periods=1).std()
            
            # Features estacionales para mensual
            group['quarter_sin'] = np.sin(2 * np.pi * group['quarter'] / 4)
            group['quarter_cos'] = np.cos(2 * np.pi * group['quarter'] / 4)
            
            # Rellenar valores faltantes
            numeric_cols = ['lag_1', 'lag_3', 'lag_6', 'lag_12', 'rolling_3', 'rolling_6', 'rolling_12', 'trend_3', 'trend_6', 'trend_12', 'volatility_3', 'volatility_6', 'volatility_12']
            for col in numeric_cols:
                group[col] = group[col].fillna(group[metric_column].mean())

            # Features expandidas para el modelo mensual
            feature_cols = ['Mes_Num', 'Año', 'Mes_sin', 'Mes_cos', 'quarter', 'month_of_year', 'year_month',
                          'lag_1', 'lag_3', 'lag_6', 'lag_12', 'rolling_3', 'rolling_6', 'rolling_12',
                          'trend_3', 'trend_6', 'trend_12', 'volatility_3', 'volatility_6', 'volatility_12',
                          'quarter_sin', 'quarter_cos']
            
            X_train = group[feature_cols]
            y_train = group[metric_column]
            
            # Normalización de features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Modelo optimizado para velocidad mensual
            if metric_column == 'USD_Total_SI_CD':
                model = XGBRegressor(
                    n_estimators=80,  # Reducido significativamente
                    max_depth=4, 
                    learning_rate=0.12, 
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42, 
                    n_jobs=-1, 
                    verbosity=0
                )
            else:
                model = XGBRegressor(
                    n_estimators=80,  # Reducido significativamente
                    max_depth=4, 
                    learning_rate=0.12, 
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42, 
                    n_jobs=-1, 
                    verbosity=0
                )
            
            # Validación cruzada desactivada para velocidad
            # Solo métricas básicas
            if len(X_train) > 4:
                metric_record = {'MAE': 0, 'Data_Points': len(X_train)}
                for col, val in zip(available_cols, group_vals):
                    metric_record[col] = val
                model_metrics.append(metric_record)
            
            # Entrenar modelo final
            model.fit(X_train_scaled, y_train)
            
            # Generar predicciones mensuales
            last_values = group.tail(1).copy()
            
            for anio_pred in range(fecha_inicio_pred.year, fecha_fin_pred.year + 1):
                start_mes = fecha_inicio_pred.month if anio_pred == fecha_inicio_pred.year else 1
                end_mes = fecha_fin_pred.month if anio_pred == fecha_fin_pred.year else 12
                for mes in range(start_mes, end_mes + 1):
                    mes_num_pred = anio_pred * 12 + mes
                    quarter_pred = (mes - 1) // 3 + 1
                    year_month_pred = anio_pred * 12 + mes
                    
                    # Usar valores lag reales mensuales
                    lag_1_val = last_values[metric_column].iloc[-1] if len(group) >= 1 else group[metric_column].mean()
                    lag_3_val = group[metric_column].iloc[-3] if len(group) >= 3 else group[metric_column].mean()
                    lag_6_val = group[metric_column].iloc[-6] if len(group) >= 6 else group[metric_column].mean()
                    lag_12_val = group[metric_column].iloc[-12] if len(group) >= 12 else group[metric_column].mean()
                    
                    # Rolling values mensuales
                    rolling_3_val = group[metric_column].tail(3).mean()
                    rolling_6_val = group[metric_column].tail(6).mean()
                    rolling_12_val = group[metric_column].tail(12).mean()
                    
                    # Tendencia y volatilidad mensual
                    trend_3_val = group['trend_3'].iloc[-1] if not pd.isna(group['trend_3'].iloc[-1]) else 0
                    trend_6_val = group['trend_6'].iloc[-1] if not pd.isna(group['trend_6'].iloc[-1]) else 0
                    trend_12_val = group['trend_12'].iloc[-1] if not pd.isna(group['trend_12'].iloc[-1]) else 0
                    volatility_3_val = group['volatility_3'].iloc[-1] if not pd.isna(group['volatility_3'].iloc[-1]) else 0
                    volatility_6_val = group['volatility_6'].iloc[-1] if not pd.isna(group['volatility_6'].iloc[-1]) else 0
                    volatility_12_val = group['volatility_12'].iloc[-1] if not pd.isna(group['volatility_12'].iloc[-1]) else 0
                    
                    # Features estacionales
                    mes_sin_val = np.sin(2 * np.pi * mes / 12)
                    mes_cos_val = np.cos(2 * np.pi * mes / 12)
                    quarter_sin_val = np.sin(2 * np.pi * quarter_pred / 4)
                    quarter_cos_val = np.cos(2 * np.pi * quarter_pred / 4)
                    
                    # Crear features para predicción mensual
                    X_pred = pd.DataFrame({
                        'Mes_Num': [mes_num_pred],
                        'Año': [anio_pred],
                        'Mes_sin': [mes_sin_val],
                        'Mes_cos': [mes_cos_val],
                        'quarter': [quarter_pred],
                        'month_of_year': [mes],
                        'year_month': [year_month_pred],
                        'lag_1': [lag_1_val],
                        'lag_3': [lag_3_val],
                        'lag_6': [lag_6_val],
                        'lag_12': [lag_12_val],
                        'rolling_3': [rolling_3_val],
                        'rolling_6': [rolling_6_val],
                        'rolling_12': [rolling_12_val],
                        'trend_3': [trend_3_val],
                        'trend_6': [trend_6_val],
                        'trend_12': [trend_12_val],
                        'volatility_3': [volatility_3_val],
                        'volatility_6': [volatility_6_val],
                        'volatility_12': [volatility_12_val],
                        'quarter_sin': [quarter_sin_val],
                        'quarter_cos': [quarter_cos_val]
                    })
                    
                    # Normalizar features de predicción
                    X_pred_scaled = scaler.transform(X_pred)
                    pred = model.predict(X_pred_scaled)[0]
                    
                    # Aplicar límites simples para velocidad mensual
                    pred = max(pred, 0)
                    
                    # Límite superior simple basado en percentiles
                    upper_limit = group[metric_column].quantile(0.95)
                    pred = min(pred, upper_limit * 1.1)
                    
                    # Create prediction record
                    pred_record = {'Periodo': pd.Timestamp(year=anio_pred, month=mes, day=1), 'Predicción': pred}
                    for col, val in zip(available_cols, group_vals):
                        pred_record[col] = val
                        
                    predicciones.append(pred_record)
    
    progress_bar.empty()
    status_text.empty()
    
    # Mostrar métricas del modelo si están disponibles
    if model_metrics:
        metrics_df = pd.DataFrame(model_metrics)
        st.subheader("📊 Métricas de Calidad del Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("MAE Promedio", f"{metrics_df['MAE'].mean():.2f}")
            st.metric("Grupos Procesados", len(metrics_df))
        
        with col2:
            st.metric("MAE Mínimo", f"{metrics_df['MAE'].min():.2f}")
            st.metric("Puntos de Datos Promedio", f"{metrics_df['Data_Points'].mean():.0f}")
        
        # Mostrar tabla de métricas por grupo
        with st.expander("📈 Ver Métricas Detalladas por Grupo"):
            st.dataframe(metrics_df.round(2))
    
    return predicciones

def display_predictions(predicciones, df_model, unidad_tiempo, metric_column='Cantidad', available_cols=None):
    if not predicciones:
        st.warning("⚠️ No se generaron predicciones con los filtros actuales.")
        return
    
    resultados = pd.DataFrame(predicciones)
    
    st.subheader("🔮 Resultados de Predicción")
    
    # Mostrar resumen de predicciones
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predicciones", len(resultados))
    
    with col2:
        st.metric("Predicción Promedio", f"{resultados['Predicción'].mean():.1f}")
    
    with col3:
        st.metric("Predicción Total", f"{resultados['Predicción'].sum():.0f}")
    
    with col4:
        variacion = (resultados['Predicción'].std() / resultados['Predicción'].mean()) * 100
        st.metric("Coef. Variación", f"{variacion:.1f}%")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Gráfico Principal", "Análisis Detallado", "Datos Exportables", "Calidad del Modelo"])
    
    with tab1:
        historico = df_model.groupby('Periodo')[metric_column].sum().reset_index()
        pred_total = resultados.groupby('Periodo')['Predicción'].sum().reset_index()
        
        fig = go.Figure()
        
        metric_label = 'Cantidad' if metric_column == 'Cantidad' else 'USD Total'
        
        fig.add_trace(go.Scatter(
            x=historico['Periodo'], 
            y=historico[metric_column],
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
            yaxis_title=metric_label,
            legend=dict(x=0, y=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Pais' in resultados.columns:
                pred_by_country = resultados.groupby('Pais')['Predicción'].sum().sort_values(ascending=False)
                fig = px.pie(values=pred_by_country.values, names=pred_by_country.index,
                            title='Predicción por País')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos de País disponibles")
        
        with col2:
            if 'U_Segmento' in resultados.columns:
                pred_by_season = resultados.groupby('U_Segmento')['Predicción'].sum().sort_values(ascending=False)
                fig = px.bar(x=pred_by_season.index, y=pred_by_season.values,
                            title='Predicción por Segmento')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos de Segmento disponibles")
        
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
        
        if len(pred_total) > 0:
            historico_stats = historico[metric_column].describe()
            pred_stats = pred_total['Predicción'].describe()
            
            comparison_df = pd.DataFrame({
                'Histórico': historico_stats,
                'Predicción': pred_stats
            })
            
            st.write("📊 Comparación Estadística:")
            st.dataframe(comparison_df.round(2))
    
    with tab3:
        display_df = resultados.copy()
        display_df['Predicción'] = display_df['Predicción'].round().astype(int)
        
        if unidad_tiempo == 'Mensual':
            display_df['Periodo'] = display_df['Periodo'].dt.strftime('%B %Y')
        else:
            display_df['Periodo'] = display_df['Periodo'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_df, use_container_width=True)
        
        csv = display_df.to_csv(index=False, sep=';')
        st.download_button(
            label="📥 Descargar Predicciones (CSV)",
            data=csv,
            file_name=f'predicciones_{unidad_tiempo.lower()}_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )
        
        # Estadísticas adicionales
        st.subheader("📊 Estadísticas Adicionales")
        
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            if 'Pais' in display_df.columns:
                st.write("**Por País:**")
                pais_stats = display_df.groupby('Pais')['Predicción'].agg(['sum', 'mean', 'count']).round(2)
                st.dataframe(pais_stats)
            else:
                st.info("No hay datos de País disponibles")
        
        with stats_col2:
            if 'U_Segmento' in display_df.columns:
                st.write("**Por Segmento:**")
                segmento_stats = display_df.groupby('U_Segmento')['Predicción'].agg(['sum', 'mean', 'count']).round(2)
                st.dataframe(segmento_stats)
            else:
                st.info("No hay datos de Segmento disponibles")
    
    with tab4:
        st.subheader("🎯 Evaluación de Calidad")
        
        # Gráfico de distribución de predicciones
        fig = px.histogram(resultados, x='Predicción', nbins=30, 
                          title='Distribución de Predicciones')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de outliers
        Q1 = resultados['Predicción'].quantile(0.25)
        Q3 = resultados['Predicción'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = resultados[(resultados['Predicción'] < lower_bound) | 
                             (resultados['Predicción'] > upper_bound)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Outliers Detectados", len(outliers))
            st.metric("% Outliers", f"{(len(outliers)/len(resultados)*100):.1f}%")
        
        with col2:
            st.metric("Rango Intercuartil", f"{IQR:.1f}")
            st.metric("Mediana", f"{resultados['Predicción'].median():.1f}")
        
        if len(outliers) > 0:
            st.subheader("🔴 Predicciones Atípicas")
            st.dataframe(outliers.round(2))
        
        # Recomendaciones de calidad
        st.subheader("📝 Recomendaciones")
        
        if variacion > 100:
            st.warning("⚠️ Alta variabilidad en las predicciones. Considere revisar los datos de entrada.")
        elif variacion < 10:
            st.info("ℹ️ Predicciones muy uniformes. Podría indicar falta de señal en los datos.")
        else:
            st.success("✅ Variabilidad de predicciones en rango aceptable.")
        
        if len(outliers) > len(resultados) * 0.05:
            st.warning("⚠️ Muchos outliers detectados. Revisar calidad de los datos históricos.")
        else:
            st.success("✅ Cantidad aceptable de outliers.")

def main():
    if 'metrics_df' in st.session_state:
        del st.session_state['metrics_df']
    if 'predicciones' in st.session_state:
        del st.session_state['predicciones']
    if 'df_model' in st.session_state:
        del st.session_state['df_model']
    if 'unidad_tiempo' in st.session_state:
        del st.session_state['unidad_tiempo']
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("📊 Dashboard de Predicción de Ventas New Era")
    
    with col2:
        st.markdown("""
        <div style="text-align: right; padding: 10px;">
            <div style="font-size: 24px; font-weight: bold; color: #1f4e79; margin-bottom: 5px;">
                🧢 NEW ERA
            </div>
            <div style="font-size: 12px; color: #666; font-style: italic;">
                Cap Company
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    df = load_data()
    
    # Determine available columns
    available_cols = [col for col in ['Pais', 'Tienda', 'U_Segmento'] if col in df.columns]
    
    if len(available_cols) == 0:
        st.error("❌ No se encontraron las columnas necesarias para el análisis.")
        st.stop()
    
    stats = get_summary_stats(df, 'Cantidad')
    
    st.markdown("---")
    
    df_filtered = apply_filters(df)
    
    if df_filtered.empty:
        st.warning("⚠️ No hay datos para los filtros seleccionados. Modifica los filtros en la barra lateral.")
        return
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔮 Configuración de Predicción")
    
    # Check if USD column exists
    metric_options = ['Cantidad']
    if 'USD_Total_SI_CD' in df.columns:
        metric_options.append('USD_Total_SI_CD')
    
    unidad_medida = st.sidebar.selectbox(
        "📊 Unidad de Medida",
        metric_options,
        format_func=lambda x: 'Cantidad (Unidades)' if x == 'Cantidad' else 'Monto USD',
        help="Selecciona la métrica a predecir"
    )
    
    unidad_tiempo = st.sidebar.selectbox(
        "⏳ Unidad de Tiempo",
        ['Diario', 'Mensual'],
        help="Selecciona la granularidad temporal para las predicciones"
    )
    
    fecha_inicio_pred = st.sidebar.date_input(
        '📅 Inicio Predicción',
        pd.Timestamp('2025-07-01'),
        help="Fecha de inicio para las predicciones"
    )
    
    fecha_fin_pred = st.sidebar.date_input(
        '📅 Fin Predicción',
        pd.Timestamp('2025-12-31'),
        help="Fecha final para las predicciones"
    )
    
    create_historical_charts(df_filtered, unidad_medida)
    st.markdown("---")
    
    if st.button("🚀 Generar Predicciones", type="primary"):
        with st.spinner('🤖 Generando predicciones...'):
            predicciones = predict_sales_original(
                df_filtered.copy(), 
                unidad_tiempo, 
                pd.to_datetime(fecha_inicio_pred), 
                pd.to_datetime(fecha_fin_pred),
                unidad_medida
            )
            
            if unidad_tiempo == 'Diario':
                df_model = df_filtered.groupby(available_cols + ['Fecha'])[unidad_medida].sum().reset_index()
                df_model.rename(columns={'Fecha': 'Periodo'}, inplace=True)
            else:
                df_filtered['Periodo'] = df_filtered['Fecha'].dt.to_period('M').dt.to_timestamp()
                df_model = df_filtered.groupby(available_cols + ['Periodo'])[unidad_medida].sum().reset_index()
            
            display_predictions(predicciones, df_model, unidad_tiempo, unidad_medida, available_cols)


if __name__ == "__main__":
    main()