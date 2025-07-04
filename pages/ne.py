import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import timedelta, datetime
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

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
    
    # Determine available grouping columns
    grouping_cols = ['Pais', 'Tienda', 'U_Segmento']
    available_cols = [col for col in grouping_cols if col in df.columns]
    
    if len(available_cols) == 0:
        st.error("No hay suficientes columnas para realizar la predicción")
        return predicciones
    
    if unidad_tiempo == 'Diario':
        peso_promedio = 0.3
        peso_modelo = 0.7

        df['weekday'] = df['Fecha'].dt.weekday
        fines_df = df[df['weekday'].isin([5, 6])].copy()
        fines_df['year'] = fines_df['Fecha'].dt.year
        fines_df = fines_df[fines_df['year'].isin([2023, 2024, 2025])]
        
        if len(available_cols) >= 3:
            promedios_fines = (
                fines_df.groupby(available_cols + ['weekday'])[metric_column]
                .mean()
                .reset_index()
                .rename(columns={metric_column: 'Promedio_Finde'})
            )
        else:
            promedios_fines = pd.DataFrame()

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
            
            group['lag_7'] = group[metric_column].shift(7)
            group['lag_30'] = group[metric_column].shift(30)
            group['rolling_7'] = group[metric_column].rolling(window=7, min_periods=1).mean()
            group['rolling_30'] = group[metric_column].rolling(window=30, min_periods=1).mean()
            
            # Pure XGBoost prediction - no artificial trends
            # Let the model learn from data naturally for both USD and Quantity
            
            group = group.fillna(group[metric_column].mean())

            # Time features that capture trends and seasonality
            X_train = group[['Dia_Num', 'weekday', 'month', 'quarter', 'year']]
            y_train = group[metric_column]

            if metric_column == 'USD_Total_SI_CD':
                model = XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.06, random_state=42, n_jobs=-1, verbosity=0)
            else:
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
                
                # Proper lag features using actual historical values
                if len(group) >= 7:
                    lag_7_val = group[metric_column].iloc[-7]
                else:
                    lag_7_val = group[metric_column].mean()
                    
                if len(group) >= 30:
                    lag_30_val = group[metric_column].iloc[-30]
                else:
                    lag_30_val = group[metric_column].mean()
                    
                rolling_7_val = group[metric_column].tail(7).mean()
                rolling_30_val = group[metric_column].tail(30).mean()

                # Prediction features including trend
                X_pred = pd.DataFrame({
                    'Dia_Num': [dia_num_pred],
                    'weekday': [weekday_pred],
                    'month': [month_pred],
                    'quarter': [quarter_pred],
                    'year': [year_pred]
                })

                pred_model = model.predict(X_pred)[0]  # Pure XGBoost prediction
                
                # Pure XGBoost prediction - no weekend adjustments
                pred = max(pred_model, 0)

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
            if group.shape[0] < 2:
                continue

            # Pure XGBoost prediction for monthly - no artificial trends
            # Let the model learn from data naturally for both USD and Quantity

            # Monthly features with trend
            X_train = group[['Mes_Num', 'Año', 'Mes_sin', 'Mes_cos']]
            y_train = group[metric_column]

            if metric_column == 'USD_Total_SI_CD':
                model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42, n_jobs=-1, verbosity=0)
            else:
                model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
            model.fit(X_train, y_train)

            for anio_pred in range(fecha_inicio_pred.year, fecha_fin_pred.year + 1):
                start_mes = fecha_inicio_pred.month if anio_pred == fecha_inicio_pred.year else 1
                end_mes = fecha_fin_pred.month if anio_pred == fecha_fin_pred.year else 12
                for mes in range(start_mes, end_mes + 1):
                    mes_num_pred = anio_pred * 12 + mes
                    # Monthly prediction with trend
                    X_pred = pd.DataFrame({
                        'Mes_Num': [mes_num_pred],
                        'Año': [anio_pred],
                        'Mes_sin': [np.sin(2 * np.pi * mes / 12)],
                        'Mes_cos': [np.cos(2 * np.pi * mes / 12)]
                    })
                    pred = model.predict(X_pred)[0]  # Pure XGBoost prediction
                    pred = max(pred, 0)
                    
                    # Create prediction record
                    pred_record = {'Periodo': pd.Timestamp(year=anio_pred, month=mes, day=1), 'Predicción': pred}
                    for col, val in zip(available_cols, group_vals):
                        pred_record[col] = val
                        
                    predicciones.append(pred_record)
    
    progress_bar.empty()
    status_text.empty()
    
    return predicciones

def display_predictions(predicciones, df_model, unidad_tiempo, metric_column='Cantidad', available_cols=None):
    if not predicciones:
        st.warning("⚠️ No se generaron predicciones con los filtros actuales.")
        return
    
    resultados = pd.DataFrame(predicciones)
    
    st.subheader("🔮 Resultados de Predicción")
    
    tab1, tab2, tab3 = st.tabs(["Gráfico Principal", "Análisis Detallado", "Datos Exportables"])
    
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