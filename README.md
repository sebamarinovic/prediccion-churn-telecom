# Predicción de Fuga de Clientes (Churn Prediction) 📉🚫

**Asignatura:** Machine Learning II  
**Tecnologías:** Python, Scikit-Learn, Pandas, Seaborn.

## 📌 Contexto del Proyecto
Una empresa de telecomunicaciones busca identificar proactivamente a los clientes con alta probabilidad de fugarse (*churn*). El objetivo de este proyecto es construir y evaluar un modelo predictivo robusto que permita al equipo de retención priorizar sus esfuerzos.

## 🚀 Metodología
El flujo de trabajo implementado incluye:
1.  **Preprocesamiento:** Limpieza de datos, imputación de nulos y codificación *One-Hot* de variables categóricas.
2.  **Ingeniería de Características:** Generación de interacciones polinomiales (grado 2) para capturar relaciones no lineales.
3.  **Selección de Modelos:** Comparación entre Regresión Logística Base vs. Polinomial.
4.  **Regularización:** Aplicación de penalización **Lasso (L1)** para selección automática de características y control del sobreajuste.
5.  **Optimización:** Búsqueda de hiperparámetros con `GridSearchCV` maximizando el **PR-AUC** (Average Precision) debido al desbalance de clases.

## 📊 Resultados Clave
* **Mejor Modelo:** Regresión Logística con características polinomiales y regularización Lasso ($C \approx 1.29$).
* **Métricas de Desempeño:**
    * **ROC-AUC:** 0.85 (Excelente discriminación).
    * **Recall (Clase Churn):** 0.80 (Aplicando balanceo de clases).
* **Hallazgo de Negocio:** Se demostró que al ajustar el peso de las clases (`class_weight='balanced'`), es posible aumentar la detección de fugas del 54% al 80%, lo cual es crítico para una estrategia de retención agresiva.

## 🛠️ Instalación y Uso
1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/TU_USUARIO/machine-learning-churn.git](https://github.com/sebamarinovic/prediccion-churn-telecom)
