##
## Aplicacion web para visualizacion del modelo entrenado de analisis de mora.

import joblib
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Configuración de página
st.set_page_config(
    page_title="Sistema Predictivo de Crisis Crediticia",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal con estilo
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('# 🧠 Sistema Predictivo de Riesgo Crediticio', unsafe_allow_html=True)
st.markdown("## Plan Avanzado: Predicciones Multi-Horizonte")

# Cargar datos y modelos mejorados
@st.cache_data
def cargar_datos_mejorados():
    df = pd.read_csv("/app/modelos_cnn/datos_dashboard_multi_18m.csv")
    df["mes"] = pd.to_datetime(df["mes"]).dt.date
    
    # Calcular métricas adicionales
    df['crecimiento_mensual'] = df.groupby(['riesgo', 'sector'])['num_creditos'].pct_change()
    df['score_riesgo'] = (df['tasa_mora_90'] * 0.4 + 
                         df['tasa_judicial'] * 0.3 + 
                         df['dias_mora_promedio'] * 0.3)
    
    return df

# Funcion para cargar el modelo, el scaler o transformador, y la configuracion que es un json  
@st.cache_resource
def cargar_modelo_multioutput():
    modelo = load_model("/app/modelos_cnn/modelo_cnn_multi_18m.h5")
    scaler = joblib.load("/app/modelos_cnn/scaler_multi_18m.pkl")
    with open("/app/modelos_cnn/config_18m.json", "r") as f:
        config = json.load(f)
    return modelo, scaler, config

# Cargar datos
df = cargar_datos_mejorados()
modelo, scaler, config = cargar_modelo_multioutput()

# Sidebar mejorado -> Este es el menu lateral para manipulacion de la data!!
st.sidebar.header("🎯 Configuración de Predicción")

# Selector de nivel de análisis, en donde index 0, indica que inicializa en "Sucursal"
nivel_analisis = st.sidebar.radio(
    "Nivel de Análisis:",
    ["Sucursal", "Sector", "Provincia", "Grupo de Riesgo"],
    index=0
)

# Filtros dinámicos según nivel -> agrego sorted, los datos estaban desordenados
if nivel_analisis == "Sucursal":
    sucursales = ['Todos'] + sorted(df['codigo_sucursal'].unique())
    filtro_principal = st.sidebar.selectbox("Sucursal", sucursales)
    
elif nivel_analisis == "Sector":
    sectores = ['Todos'] + list(sorted(df['sector'].unique()))
    filtro_principal = st.sidebar.selectbox("Sector Económico", sectores)
    
elif nivel_analisis == "Provincia":
    provincias = ['Todos'] + list(sorted(df['codigo_provincia'].unique()))
    filtro_principal = st.sidebar.selectbox("Provincia", provincias)
    
else:  # Grupo de Riesgo
    riesgos = ['Todos'] + list(sorted(df['riesgo'].unique()))
    filtro_principal = st.sidebar.selectbox("Nivel de Riesgo", riesgos)

# Datos para las barras de prediccion
# Selector de horizonte temporal
horizonte_pred = st.sidebar.select_slider(
    "Horizonte de Predicción (meses):",
    options=[1, 3, 6, 12, 18],
    value=6
)

# Datos para la barra de meses de historial.
# Número de meses históricos a visualizar
meses_historia = st.sidebar.slider("Meses de historia a mostrar:", 3, 24, 12)

# Función para preparar datos de predicción
def preparar_datos_prediccion(df_filtrado, horizonte):
    """
    Prepara secuencias para predicción multi-horizonte
    """
    # Obtener los últimos 6 meses para predicción
    ultimos_meses = df_filtrado.sort_values('mes').tail(6)
    
    if len(ultimos_meses) < 6:
        return None
    
    # Seleccionar features
    features = config['features_numericas']
    X_pred = ultimos_meses[features].values
    
    # Escalar: aplicar scaler por fila (cada timestep tiene 'len(features)' columnas)
    # X_pred tiene forma (6, n_features). scaler fue entrenado sobre arrays (n_rows, n_features).
    X_scaled = scaler.transform(X_pred).reshape(1, 6, len(features)).astype(np.float32)
    
    return X_scaled

# Función de predicción
def realizar_prediccion(X_input, horizonte):
    """
    Realiza predicción para el horizonte seleccionado
    """
    if X_input is None:
        return None
    # Obtener todas las predicciones
    predicciones = modelo.predict(X_input, verbose=0)

    # Intentar mapeo por nombre (compatibilidad con versiones previas)
    horizonte_map = {
        1: 'pred_1m',
        3: 'pred_3m',
        6: 'pred_6m',
        12: 'pred_12m',
        18: 'pred_18m'
    }

    output_names = list(getattr(modelo, 'output_names', []))
    desired_name = horizonte_map.get(horizonte)
    output_index = None

    if desired_name and desired_name in output_names:
        output_index = output_names.index(desired_name)
    else:
        # Fallback: asumir que las salidas son secuenciales por horizonte (horizonte_1 .. horizonte_N)
        try:
            idx = int(horizonte) - 1
            if idx < 0:
                raise IndexError
            if idx < len(predicciones):
                output_index = idx
            else:
                # Si el horizonte pedido excede salidas, usar la última salida disponible
                output_index = max(0, len(predicciones) - 1)
        except Exception:
            # Último intento: buscar nombres que empiecen por 'horizonte_{horizonte}'
            for i, name in enumerate(output_names):
                if name.startswith(f'horizonte_{horizonte}'):
                    output_index = i
                    break

    if output_index is None:
        st.error(f"No se pudo mapear el horizonte {horizonte} a una salida del modelo")
        return None

    # Asegurar que accedemos correctamente al array de predicción
    pred_array = predicciones[output_index]
    probabilidad = float(np.asarray(pred_array).flatten()[0] * 100)
    return probabilidad

# Aplicar filtros
df_filtrado = df.copy()
if filtro_principal != 'Todos':
    if nivel_analisis == "Sucursal":
        df_filtrado = df_filtrado[df_filtrado['codigo_sucursal'] == filtro_principal]
    elif nivel_analisis == "Sector":
        df_filtrado = df_filtrado[df_filtrado['sector'] == filtro_principal]
    elif nivel_analisis == "Provincia":
        df_filtrado = df_filtrado[df_filtrado['codigo_provincia'] == filtro_principal]
    else:
        df_filtrado = df_filtrado[df_filtrado['riesgo'] == filtro_principal]

# PRIMERA FILA: Métricas Principales
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_creditos = df_filtrado['num_creditos'].sum()
    st.metric("📊 Total Créditos", f"{total_creditos:n}")

with col2:
    tasa_crisis = df_filtrado['crisis_flag'].mean() * 100
    st.metric("⚠️ Tasa Crisis Actual", f"{tasa_crisis:.1f}%")

with col3:
    # Preparar y realizar predicción
    X_pred = preparar_datos_prediccion(df_filtrado, horizonte_pred)
    prob_crisis = realizar_prediccion(X_pred, horizonte_pred)
    
    if prob_crisis is not None:
        st.metric(
            f"🔮 Predicción {horizonte_pred} meses", 
            f"{prob_crisis:.1f}%",
            delta=f"{(prob_crisis - tasa_crisis):+.1f}%"
        )
    else:
        st.metric(f"Predicción {horizonte_pred} meses", "Datos insuficientes")

with col4:
    mora_promedio = df_filtrado['tasa_mora_90'].mean()
    st.metric("⏰ Mora >90d", f"{mora_promedio:.1f}%")

with col5:
    accuracy_modelo = config['metricas_finales']['accuracy'] * 100
    st.metric("🎯 Accuracy Modelo", f"{accuracy_modelo:.1f}%")

# SEGUNDA FILA: Predicciones Multi-Horizonte
st.subheader("📈 Predicciones de Riesgo por Horizonte Temporal")

# Calcular predicciones para todos los horizontes
horizontes = [1, 3, 6, 12, 18]
predicciones_horizontes = []

for h in horizontes:
    X_pred_h = preparar_datos_prediccion(df_filtrado, h)
    prob_h = realizar_prediccion(X_pred_h, h) if X_pred_h is not None else 0
    predicciones_horizontes.append(prob_h)

# Gráfico de predicciones multi-horizonte
fig_horizontes = go.Figure()

fig_horizontes.add_trace(go.Scatter(
    x=horizontes,
    y=predicciones_horizontes,
    mode='lines+markers+text',
    name='Probabilidad de Crisis',
    line=dict(color='red', width=3),
    marker=dict(size=10),
    text=[f'  {round(p, 2)}%' for p in predicciones_horizontes],
    textposition='top center'
))

fig_horizontes.update_layout(
    title=f'Predicción de Riesgo a Diferentes Horizontes - {nivel_analisis}: {filtro_principal}',
    xaxis_title='Horizonte (meses)',
    yaxis_title='Probabilidad de Crisis (%)',
    yaxis_range=[0, 100],
    height=400,
    template='plotly_white'
)

st.plotly_chart(fig_horizontes, use_container_width=True)

# TERCERA FILA: Gráficos Comparativos
col1, col2 = st.columns(2)

with col1:
    # Evolución histórica vs predicción
    ultimos_meses = df_filtrado.sort_values('mes').tail(meses_historia)
    
    fig_evolucion = go.Figure()
    
    # Datos históricos
    fig_evolucion.add_trace(go.Scatter(
        x=ultimos_meses['mes'],
        y=ultimos_meses['crisis_flag'] * 100,
        mode='lines+markers',
        name='Crisis Histórica',
        line=dict(color='blue', width=2)
    ))
    
    # Predicción futura
    fecha_ultima = ultimos_meses['mes'].max()
    fechas_futuras = [fecha_ultima + pd.DateOffset(months=i) for i in range(1, horizonte_pred + 1)]
    
    fig_evolucion.add_trace(go.Scatter(
        x=fechas_futuras,
        y=[predicciones_horizontes[horizontes.index(horizonte_pred)]] * horizonte_pred,
        mode='lines+markers',
        name=f'Predicción {horizonte_pred} meses',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig_evolucion.update_layout(
        title='Evolución Histórica vs Predicción Futura',
        height=400
    )
    
    st.plotly_chart(fig_evolucion, use_container_width=True)

with col2:
    # Análisis de factores de riesgo
    factores_df = pd.DataFrame({
        'Factor': ['Mora >90 días', 'Procesos Judiciales', 'Días Mora Promedio', 
                  'Gestión Cobro', 'Rotación Cartera'],
        'Peso': [0.4, 0.3, 0.2, 0.05, 0.05],
        'Valor Actual': [
            df_filtrado['tasa_mora_90'].mean(),
            df_filtrado['tasa_judicial'].mean(),
            df_filtrado['dias_mora_promedio'].mean(),
            (df_filtrado['total_gestion_cobro'].sum() / df_filtrado['monto_total'].sum() * 100),
            (df_filtrado['creditos_cerrados'].sum() / df_filtrado['num_creditos'].sum() * 100)
        ]
    })
    
    fig_factores = px.bar(
        factores_df,
        x='Factor',
        y='Valor Actual',
        color='Peso',
        title='Análisis de Factores de Riesgo',
        color_continuous_scale='RdYlGn_r',
        height=400
    )
    
    st.plotly_chart(fig_factores, use_container_width=True)

# CUARTA FILA: Recomendaciones y Alertas
st.subheader("🚨 Recomendaciones y Alertas")

# Generar recomendaciones basadas en predicciones
if predicciones_horizontes[0] > 30:  # Predicción a 1 mes > 30%
    st.error(f"**ALERTA INMINENTE**: Probabilidad de crisis del {predicciones_horizontes[0]:.1f}% en el próximo mes")
    st.info("**Recomendaciones inmediatas:**")
    st.markdown("""
    1. Revisar cartera de mayor riesgo
    2. Aumentar seguimiento a créditos con mora > 60 días
    3. Evaluar provisiones necesarias
    """)

elif predicciones_horizontes[2] > 40:  # Predicción a 6 meses > 40%
    st.warning(f"**ALERTA MEDIO PLAZO**: Probabilidad de crisis del {predicciones_horizontes[2]:.1f}% en 6 meses")
    st.info("**Acciones recomendadas:**")
    st.markdown("""
    1. Reestructurar políticas de crédito
    2. Implementar modelos de scoring más estrictos
    3. Diversificar cartera por sectores
    """)

else:
    st.success("**SITUACIÓN ESTABLE**: Probabilidades de crisis por debajo de umbrales críticos")
    st.info("**Mantenimiento preventivo:**")
    st.markdown("""
    1. Continuar monitoreo regular
    2. Mantener políticas actuales
    3. Actualizar modelos mensualmente
    """)

# QUINTA FILA: Reporte Detallado
st.subheader("Reporte Detallado y Datos de Análisis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Predicciones por Horizonte")
    pred_df = pd.DataFrame({
        'Horizonte (meses)': horizontes,
        'Probabilidad Crisis (%)':  predicciones_horizontes,
        'Nivel Alerta': ['Alto' if p > 30 else 'Medio' if p > 15 else 'Bajo' 
                        for p in predicciones_horizontes]
    })
    st.dataframe(pred_df.style.background_gradient(subset=['Probabilidad Crisis (%)'], 
                                                    cmap='RdYlGn_r'))

with col2:
    st.markdown("### Métricas Clave Actuales")
    metricas_actuales = {
        'Total Créditos Activos': int(df_filtrado['num_creditos'].sum()),
        'Monto Total Cartera': f"${df_filtrado['monto_total'].sum():,.0f}",
        'Tasa Morosidad (>90d)': f"{df_filtrado['tasa_mora_90'].mean():.2f}%",
        'Créditos en Proceso Judicial': f"{df_filtrado['tasa_judicial'].mean():.2f}%",
        'Eficiencia Cobranza': f"${df_filtrado['total_gestion_cobro'].sum():,.0f}",
        'Score Riesgo Promedio': f"{df_filtrado['score_riesgo'].mean():.2f}/100"
    }
    
    for key, value in metricas_actuales.items():
        st.markdown(f"**{key}:** {value}")

# Datos brutos (opcional)
if st.checkbox("Mostrar datos detallados"):
    st.dataframe(df_filtrado)
        
st.markdown('---')
st.markdown('**Aprendizaje Profundo**') 
st.markdown('**Profesor**: Phd. Oscar Chang**')
