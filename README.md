# Predicción de Fuga de Clientes (Telecom Churn Prediction) 📡

Este repositorio contiene la solución a la **Actividad 1** de la asignatura **Machine Learning II**. El proyecto consiste en la construcción y evaluación de modelos de clasificación supervisada para predecir la probabilidad de que un cliente abandone una empresa de telecomunicaciones ("Churn").

## 📄 Contexto del Problema

Una empresa de telecomunicaciones busca identificar tempranamente a los clientes con alto riesgo de fuga para diseñar campañas de retención efectivas. Se trabajó con un dataset histórico que incluye características demográficas, de servicios contratados y de facturación.

* **Variable Objetivo:** `Churn` (1 = Se va, 0 = Se queda).
* **Desafío Principal:** El desbalance de clases (la tasa de fuga es aprox. 26.5%) y la necesidad de capturar relaciones no lineales entre variables.

## 🛠️ Tecnologías Utilizadas

El proyecto fue desarrollado en **Python** utilizando el siguiente stack tecnológico:

* **Pandas & NumPy:** Limpieza, manipulación de datos e ingeniería de características.
* **Scikit-learn:**
    * *Preprocesamiento:* `StandardScaler`, `OneHotEncoder`, `SimpleImputer`.
    * *Modelado:* `LogisticRegression`.
    * *Ingeniería de Features:* `PolynomialFeatures`.
    * *Selección de Modelos:* `GridSearchCV`, `StratifiedKFold`.
    * *Métricas:* AUC-ROC, Precision-Recall, F1-Score, Matriz de Confusión.
* **Matplotlib & Seaborn:** Visualización de datos y evaluación de modelos.

## 🚀 Metodología

El flujo de trabajo implementado se divide en 4 etapas principales:

1.  **Preprocesamiento y Limpieza:**
    * Tratamiento de valores nulos en variables numéricas (`TotalCharges`).
    * Codificación de variables categóricas mediante `OneHotEncoder`.
    * Estandarización de variables numéricas.

2.  **Modelado Iterativo:**
    * **Modelo Base:** Regresión Logística estándar sin penalización.
    * **Modelo Polinomial:** Generación de interacciones de grado 2 para capturar relaciones complejas entre variables numéricas.
    * **Modelo Regularizado:** Aplicación de penalizaciones **L1 (Lasso)** y **L2 (Ridge)** para controlar la complejidad y evitar el sobreajuste.

3.  **Optimización:**
    * Uso de `GridSearchCV` para encontrar el hiperparámetro óptimo de regularización ($C$).

## 📊 Resultados Clave

El mejor modelo seleccionado fue la **Regresión Logística con Transformaciones Polinomiales y Regularización L1 (Lasso)**.

| Métrica | Resultado | Interpretación |
| :--- | :--- | :--- |
| **AUC-ROC** | **0.8475** | Alta capacidad del modelo para distinguir entre clientes que se van y los que se quedan. |
| **Accuracy** | 0.81 | Un buen rendimiento general, aunque influenciado por el desbalance de clases. |
| **Recall (Churn)**| 0.54 | Se logra detectar al 54% del total de fugas reales. |
| **Precision** | 0.67 | De los clientes alertados como riesgo, el 67% efectivamente abandonó la empresa. |

### 💡 Análisis de Regularización
La selección automática de la penalización **L1 (Lasso)** fue crítica para el éxito del modelo. De las 36 variables generadas (por los polinomios), **el modelo eliminó automáticamente 6 variables** (asignándoles coeficiente 0), reduciendo la complejidad en un **16.7%** y filtrando el ruido.

## 📂 Estructura del Repositorio

* `actividad1_ML2.ipynb`: Jupyter Notebook con el código completo, desde la carga de datos hasta la evaluación final.
* `data-churn.csv`: Dataset utilizado.
* `README.md`: Documentación del proyecto.

## 💻 Instrucciones de Ejecución

1. Clonar este repositorio.
2. Instalar las dependencias necesarias:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
