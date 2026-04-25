# train.py
# Script generado a partir del notebook Final_V2.ipynb
# Contiene el pipeline principal para entrenamiento del modelo CNN multi-horizonte

import os
import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from prometheus_client import start_http_server


# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de conexión a la base de datos
DB_CONFIG = {
    "host": "192.168.0.97",
    "port": "5432",
    "database": "analisis_db",
    "user": "usuario",
    "password": "mi_clave_segura",
}
connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
engine = create_engine(connection_string)

# Query principal para extraer datos agregados por mes-riesgo-sector
query_bloques_18m = """
WITH datos_mensuales AS (
    SELECT 
        DATE_TRUNC('month', cp.fecha_credito) as mes,
        COALESCE(cp.codigo_riesgo, 'SIN_RIESGO') as riesgo,
        COALESCE(cp.act_economica_nvl1, 'SIN_SECTOR') as sector,
        cp.codigo_provincia,
        cp.codigo_sucursal,
        COUNT(*) as num_creditos,
        SUM(cp.monto_acreditado) as monto_total,
        AVG(cp.monto_acreditado) as monto_promedio,
        AVG(cp.tot_dias_mora) as dias_mora_promedio,
        AVG(cp.tot_num_moras) as num_moras_promedio,
        COUNT(CASE WHEN cp.tot_dias_mora > 90 THEN 1 END) as creditos_mora_90,
        COUNT(CASE WHEN cp.judicial = 'S' THEN 1 END) as creditos_judiciales,
        SUM(cp.gestion_cobro) as total_gestion_cobro,
        SUM(cp.costo_judicial) as total_costo_judicial,
        AVG(cp.tasa_interes) as tasa_interes_promedio,
        AVG(COALESCE(cp.saldo_capital, 0)) as saldo_promedio,
        COUNT(CASE WHEN cp.estado_cred IN ('C', 'L') THEN 1 END) as creditos_cerrados,
        COUNT(DISTINCT cp.codigo_socio) as num_clientes_unicos,
        EXTRACT(MONTH FROM cp.fecha_credito) as mes_del_ano,
        AVG(cp.num_cuotas) as plazo_promedio,
        STDDEV(cp.monto_acreditado) as desviacion_montos,
        AVG(EXTRACT(MONTH FROM AGE(CURRENT_DATE, cp.fecha_credito))) as antiguedad_promedio_meses
    FROM cabecera_prestamos cp
    WHERE cp.fecha_credito >= '2015-07-01' 
      AND cp.fecha_credito < '2025-07-01'
    GROUP BY DATE_TRUNC('month', cp.fecha_credito), cp.codigo_riesgo, cp.act_economica_nvl1, 
             cp.codigo_provincia, cp.codigo_sucursal, EXTRACT(MONTH FROM cp.fecha_credito)
),
datos_con_lag AS (
    SELECT 
        *,
        LAG(num_creditos, 1) OVER (
            PARTITION BY riesgo, sector 
            ORDER BY mes
        ) as num_creditos_mes_anterior,
        LAG(monto_total, 1) OVER (
            PARTITION BY riesgo, sector 
            ORDER BY mes
        ) as monto_mes_anterior
    FROM datos_mensuales
)
SELECT 
    mes,
    riesgo,
    sector,
    codigo_provincia,
    codigo_sucursal,
    num_creditos,
    monto_total,
    monto_promedio,
    dias_mora_promedio,
    num_moras_promedio,
    ROUND((creditos_mora_90::numeric / NULLIF(num_creditos, 0)) * 100, 2) as tasa_mora_90,
    ROUND((creditos_judiciales::numeric / NULLIF(num_creditos, 0)) * 100, 2) as tasa_judicial,
    ROUND((creditos_cerrados::numeric / NULLIF(num_creditos, 0)) * 100, 2) as tasa_cierre,
    total_gestion_cobro,
    total_costo_judicial,
    tasa_interes_promedio,
    saldo_promedio,
    creditos_cerrados,
    num_clientes_unicos,
    ROUND(num_creditos::numeric / NULLIF(num_clientes_unicos, 0), 2) as creditos_por_cliente,
    mes_del_ano,
    ROUND(plazo_promedio::numeric, 2) as plazo_promedio,
    ROUND(desviacion_montos::numeric, 2) as desviacion_montos,
    ROUND((desviacion_montos::numeric / NULLIF(monto_promedio, 0)) * 100, 2) as coef_variacion_montos,
    ROUND(antiguedad_promedio_meses::numeric, 2) as antiguedad_promedio_meses,
    num_creditos_mes_anterior,
    ROUND(((num_creditos::numeric - COALESCE(num_creditos_mes_anterior, num_creditos)) / 
           NULLIF(num_creditos_mes_anterior, 0)) * 100, 2) as tasa_crecimiento_creditos,
    monto_mes_anterior,
    ROUND(((monto_total::numeric - COALESCE(monto_mes_anterior, monto_total)) / 
           NULLIF(monto_mes_anterior, 0)) * 100, 2) as tasa_crecimiento_monto
FROM datos_con_lag
WHERE num_creditos >= 10
ORDER BY mes, riesgo, sector
"""


def ejecutar_query(query, descripcion="", chunksize=None):
    """"""
    try:
        if chunksize:
            logging.info(
                f"Procesando {descripcion} en lotes de {chunksize:,} registros..."
            )
            dfs = []
            for chunk in pd.read_sql_query(query, engine, chunksize=chunksize):
                dfs.append(chunk)
                # Genera log
                # logging.info(
                #    f"  Lote procesado: {len(chunk):,} registros (total acumulado: {sum(len(df) for df in dfs):,})"
                # )
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_sql_query(query, engine)
        logging.info(f"{descripcion}: {len(df):,} registros obtenidos")
        return df
    except Exception as e:
        logging.error(f"Error en query {descripcion}: {e}")
        return None


def calcular_crisis_flag_mejorado(row):
    crisis_score = 0
    if row["tasa_mora_90"] > 15:
        crisis_score += 3
    if row["tasa_judicial"] > 5:
        crisis_score += 2
    if row["dias_mora_promedio"] > 45:
        crisis_score += 2
    if row["total_gestion_cobro"] > row["monto_total"] * 0.08:
        crisis_score += 1
    if row["creditos_cerrados"] / row["num_creditos"] > 0.3:
        crisis_score += 1
    if row["num_creditos"] < 50 and row["tasa_mora_90"] > 20:
        crisis_score += 2
    if row["creditos_por_cliente"] > 3:
        crisis_score += 1
    if row["tasa_crecimiento_creditos"] < -20:
        crisis_score += 1
    if row["coef_variacion_montos"] > 100:
        crisis_score += 1
    if row["plazo_promedio"] > 36 and row["tasa_mora_90"] > 10:
        crisis_score += 1
    if row["antiguedad_promedio_meses"] > 60 and row["tasa_judicial"] > 3:
        crisis_score += 1
    return 1 if crisis_score >= 4 else 0


def crear_secuencias_cnn_multi(
    df, bloque_id, features, target, ventana=6, max_horizonte=18
):
    """
    Genera secuencias temporales para entrenamiento de una CNN multi-horizonte.
    """
    df_bloque = df[df["bloque_id"] == bloque_id].sort_values("mes")
    if len(df_bloque) < ventana + max_horizonte:
        return None, None
    X_sequences = []
    y_sequences = []
    for i in range(len(df_bloque) - ventana - max_horizonte + 1):
        X_seq = df_bloque[features].iloc[i : i + ventana].values
        y_seq = []
        for h in range(1, max_horizonte + 1):
            if i + ventana + h - 1 < len(df_bloque):
                y_val = df_bloque[target].iloc[i + ventana + h - 1]
                y_seq.append(y_val)
            else:
                y_seq.append(0)
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
    return np.array(X_sequences), np.array(y_sequences)


def crear_modelo_cnn_multioutput(input_shape, num_horizontes):
    """
    Crea un modelo CNN con múltiples salidas para predecir crisis en varios horizontes de tiempo.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(
        inputs
    )
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = []
    for i in range(num_horizontes):
        output = layers.Dense(1, activation="sigmoid", name=f"horizonte_{i + 1}")(x)
        outputs.append(output)
    model = Model(inputs=inputs, outputs=outputs)
    metrics_dict = {
        f"horizonte_{i + 1}": [
            "accuracy",
            tf.keras.metrics.Precision(name=f"precision_{i + 1}"),
            tf.keras.metrics.Recall(name=f"recall_{i + 1}"),
        ]
        for i in range(num_horizontes)
    }
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=metrics_dict)
    return model


def preprocesar_y_features(df_bloques_18m):
    """
    Preprocesa los datos y crea las features necesarias para el modelo CNN multi-horizonte a 18 meses
    """
    df_features = df_bloques_18m.copy()
    df_features["mes"] = pd.to_datetime(df_features["mes"])
    df_features["bloque_id"] = (
        df_features["riesgo"]
        + "_"
        + df_features["sector"]
        + "_"
        + df_features["codigo_sucursal"].astype(str)
    )
    df_features["tasa_crecimiento_creditos"] = df_features[
        "tasa_crecimiento_creditos"
    ].fillna(0)
    df_features["tasa_crecimiento_monto"] = df_features[
        "tasa_crecimiento_monto"
    ].fillna(0)
    df_features["num_creditos_mes_anterior"] = df_features[
        "num_creditos_mes_anterior"
    ].fillna(df_features["num_creditos"])
    df_features["monto_mes_anterior"] = df_features["monto_mes_anterior"].fillna(
        df_features["monto_total"]
    )
    df_features["desviacion_montos"] = df_features["desviacion_montos"].fillna(0)
    df_features["coef_variacion_montos"] = df_features["coef_variacion_montos"].fillna(
        0
    )
    features_numericas = [
        "num_creditos",
        "monto_promedio",
        "dias_mora_promedio",
        "tasa_mora_90",
        "tasa_judicial",
        "tasa_cierre",
        "total_gestion_cobro",
        "tasa_interes_promedio",
        "creditos_por_cliente",
        "mes_del_ano",
        "plazo_promedio",
        "desviacion_montos",
        "coef_variacion_montos",
        "antiguedad_promedio_meses",
        "tasa_crecimiento_creditos",
        "tasa_crecimiento_monto",
    ]
    df_features["crisis_flag"] = df_features.apply(
        calcular_crisis_flag_mejorado, axis=1
    )
    return df_features, features_numericas


def generar_secuencias(df_features, features_numericas):
    """
    Genera las secuencias de datos para entrenar la CNN multi-horizonte a 18 meses
    """
    X_all = []
    y_all = []
    bloques_validos = []
    VENTANA_CNN = 6
    MAX_HORIZONTE = 18
    for bloque in df_features["bloque_id"].unique():
        X_seq, y_seq = crear_secuencias_cnn_multi(
            df_features,
            bloque,
            features_numericas,
            "crisis_flag",
            VENTANA_CNN,
            MAX_HORIZONTE,
        )
        if X_seq is not None and len(X_seq) > 0:
            X_all.extend(X_seq)
            y_all.extend(y_seq)
            bloques_validos.append(bloque)
    X_cnn = np.array(X_all)
    y_cnn = np.array(y_all)
    return X_cnn, y_cnn, bloques_validos, VENTANA_CNN, MAX_HORIZONTE


def normalizar_y_split(X_cnn, y_cnn, MAX_HORIZONTE):
    """Normaliza las features y divide los datos en conjuntos de entrenamiento y prueba para la CNN multi-horizonte a 18 meses"""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_cnn.reshape(-1, X_cnn.shape[-1])).reshape(
        X_cnn.shape
    )
    split_idx = int(len(X_scaled) * 0.7)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train_list = [y_cnn[:split_idx, i] for i in range(MAX_HORIZONTE)]
    y_test_list = [y_cnn[split_idx:, i] for i in range(MAX_HORIZONTE)]
    for i in range(len(y_train_list)):
        arr = np.asarray(y_train_list[i])
        y_train_list[i] = arr.reshape(
            -1,
        )
    for i in range(len(y_test_list)):
        arr = np.asarray(y_test_list[i])
        y_test_list[i] = arr.reshape(
            -1,
        )
    n_X = X_train.shape[0]
    lengths = [arr.shape[0] for arr in y_train_list]
    min_len = min([n_X] + lengths)
    if min_len < n_X or any(l != n_X for l in lengths):
        X_train = X_train[:min_len]
        y_train_list = [arr[:min_len] for arr in y_train_list]
    return X_train, X_test, y_train_list, y_test_list, scaler


def construir_modelo(X_train, MAX_HORIZONTE):
    """Construye el modelo CNN multi-horizonte a 18 meses"""
    input_shape = X_train.shape[1:]
    modelo_cnn = crear_modelo_cnn_multioutput(input_shape, MAX_HORIZONTE)
    os.makedirs("modelos_cnn", exist_ok=True)
    return modelo_cnn


def entrenar_modelo(modelo_cnn, X_train, y_train_list):
    """Entrena el modelo CNN multi-horizonte a 18 meses con early stopping y guardado del mejor modelo"""
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(
            "/app/modelos_cnn/best_model_multi_18m.h5",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    val_frac = 0.2
    n_samples = X_train.shape[0]
    val_size = int(n_samples * val_frac)
    train_end = n_samples - val_size
    X_train_final = X_train[:train_end]
    X_val = X_train[train_end:]
    y_train_final = [arr[:train_end] for arr in y_train_list]
    y_val = [arr[train_end:] for arr in y_train_list]
    historia = modelo_cnn.fit(
        X_train_final,
        y_train_final,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=0, ## si verbose esta en 1, genera mucho log y me da problemas en grafana
    )
    return historia


def evaluar_y_guardar(
    modelo_cnn,
    X_test,
    y_test_list,
    scaler,
    df_features,
    VENTANA_CNN,
    MAX_HORIZONTE,
    features_numericas,
    bloques_validos,
):
    """Evalúa el modelo CNN multi-horizonte a 18 meses en el conjunto de prueba y guarda los artefactos necesarios para producción"""
    y_pred_proba_list = modelo_cnn.predict(X_test)
    y_pred_1m = (y_pred_proba_list[0] > 0.5).astype(int).flatten()
    test_acc = accuracy_score(y_test_list[0], y_pred_1m)
    test_prec = precision_score(y_test_list[0], y_pred_1m, zero_division=0)
    test_recall = recall_score(y_test_list[0], y_pred_1m, zero_division=0)
    logging.info(f"Accuracy: {test_acc:.4f}")
    logging.info(f"Precision: {test_prec:.4f}")
    logging.info(f"Recall: {test_recall:.4f}")
    if len(np.unique(y_test_list[0])) > 1:
        try:
            auc_score = roc_auc_score(y_test_list[0], y_pred_proba_list[0].flatten())
            logging.info(f"AUC-ROC: {auc_score:.4f}")
        except Exception as e:
            logging.error(f"No se pudo calcular AUC-ROC: {e}")
    logging.info("\nReporte detallado:")
    ## carga muchos log
    ## logging.info(classification_report(y_test_list[0], y_pred_1m))
    classification_report(y_test_list[0], y_pred_1m)
    logging.info("\nMatriz de confusion:")
    ## logging.info(confusion_matrix(y_test_list[0], y_pred_1m))
    confusion_matrix(y_test_list[0], y_pred_1m)
    ## modelo_cnn.save("modelos_cnn/modelo_cnn_multi_18m", save_format="tf")
    modelo_cnn.save('/app/modelos_cnn/modelo_cnn_multi_18m.h5')
    
    logging.info("Modelo CNN Multioutput guardado")
    joblib.dump(scaler, "/app/modelos_cnn/scaler_multi_18m.pkl")
    logging.info("Scaler guardado")
    df_features.to_csv("/app/modelos_cnn/datos_dashboard_multi_18m.csv", index=False)
    logging.info("Datos dashboard guardados")
    config_18m = {
        "ventana_cnn": VENTANA_CNN,
        "max_horizonte": MAX_HORIZONTE,
        "features_numericas": features_numericas,
        "bloques_validos": bloques_validos,
        "metricas_finales": {
            "accuracy": test_acc,
            "precision": test_prec,
            "recall": test_recall,
        },
    }
    with open("/app/modelos_cnn/config_18m.json", "w") as f:
        json.dump(config_18m, f, indent=2, default=str)
    logging.info("Configuracion guardada")


def main():
    # 1. Extraer datos
    df_bloques_18m = ejecutar_query(
        query_bloques_18m,
        "Datos agregados 10 años (Jul 2015 - Jun 2025)",
        chunksize=100000,
    )
    if df_bloques_18m is None:
        logging.error("No se pudo extraer datos. Abortando.")
        return

    # 2. Preprocesar y crear features
    df_features, features_numericas = preprocesar_y_features(df_bloques_18m)

    # 3. Generar secuencias para la CNN
    X_cnn, y_cnn, bloques_validos, VENTANA_CNN, MAX_HORIZONTE = generar_secuencias(
        df_features, features_numericas
    )

    # 4. Normalizar y dividir datos
    X_train, X_test, y_train_list, y_test_list, scaler = normalizar_y_split(
        X_cnn, y_cnn, MAX_HORIZONTE
    )

    # 5. Construir el modelo
    modelo_cnn = construir_modelo(X_train, MAX_HORIZONTE)

    # 6. Entrenar el modelo
    historia = entrenar_modelo(modelo_cnn, X_train, y_train_list)

    # 7. Evaluar y guardar artefactos
    evaluar_y_guardar(
        modelo_cnn,
        X_test,
        y_test_list,
        scaler,
        df_features,
        VENTANA_CNN,
        MAX_HORIZONTE,
        features_numericas,
        bloques_validos,
    )

if __name__ == "__main__":
    start_http_server(8001)  # o 8002 según el servicio
    main()
