# Clasificación de Prendas de Ropa — Fashion-MNIST

Trabajo Práctico 02 de la materia **Laboratorio de Datos** (FCEN, UBA)

## Descripción General

Este proyecto aborda el desarrollo y la evaluación de modelos de aprendizaje supervisado sobre el dataset Fashion-MNIST. 

A través de un análisis exploratorio espacial por píxeles (utilizando métricas como el Rango Intercuartílico - IQR), se diseñaron estrategias de extracción de características para implementar modelos de clasificación binaria y multiclase, optimizando sus hiperparámetros mediante validación cruzada k-fold y evaluando su generalización en conjuntos held-out.

## Contenido y Metodología

- **Análisis Exploratorio y Pixel-wise EDA:** Inspección visual de imágenes, análisis de dispersión espacial por clase mediante heatmaps de IQR y selección orientada de atributos (píxeles de mayor varianza).
- **Clasificación Binaria (Clases 0 vs 8):** Implementación del algoritmo K-Nearest Neighbors (KNN), evaluando la influencia del número de vecinos (\(k\)) y la combinación de atributos sobre métricas como Accuracy y F1-Score.
- **Clasificación Multiclase (10 Clases):** Entrenamiento de Árboles de Decisión para la categorización completa del dataset.
- **Validación Cruzada y Selección de Hiperparámetros:** Optimización de la profundidad máxima (`max_depth`) mediante \(k\)-fold cross-validation para prevenir sobreajuste.
- **Evaluación de Modelos:** Análisis de rendimiento en el conjunto de prueba final (*held-out*) a través de matrices de confusión y reporte de métricas por clase.

## Fuente de Datos

El análisis utiliza el dataset público **Fashion-MNIST**, compuesto por:

- **70.000 imágenes** en escala de grises de \(28 \times 28\) píxeles.
- **10 categorías** de prendas de ropa y calzado (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag y Ankle boot).

## Tecnologías Utilizadas

- **Lenguaje:** Python
- **Machine Learning & Métricas:** Scikit-Learn
- **Procesamiento de Datos:** Pandas, NumPy
- **Visualización:** Matplotlib, Seaborn
- **Entorno de Desarrollo:** Jupyter Notebook / Spyder
