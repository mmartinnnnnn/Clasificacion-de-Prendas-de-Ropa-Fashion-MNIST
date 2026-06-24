"""

Laboratorio de Datos - TP2 - Clasificación y selección de modelos

        Grupo Marcel Mango

        Martín Rabinotivz
        Lucas Ballester
        Lautaro Paz Curtet
        Renato Trucillo Biglieri
        
Descripción:
    
    El código se encuentra estructurado de la siguiente manera:
        - sección con la importación de las bibliotecas utilizadas
        - Carga del dataset fashion-MNIST
        - funciones auxiliares utilizadas en las soluciones
        - Código para resolver el punto de Análisis exploratorio de los contenidos 
          del dataset
        - Sección Clasificación Binaria con train-test y knn, donde se encuentra
          el código relacionado a las soluciones del segundo punto
        - Sección Clasificación multiclase, k-folding con arboles de decisión
          con el código que realiza los experimentos del tercer punto del trabajo práctico

"""


#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

#%% Cargar datos

df = pd.read_csv("Fashion-MNIST.csv")
y = df['label']
primera_col = df.columns[0]
X = df.drop(columns=[primera_col,'label'])
#Cada fila es una imagen y cada columna es el valor del pixel en esa imagen


#%% funciones auxiliares

#genera los heatmaps de IQR por pixel
def visualizarIQRPorPixel(X,filename = ""):
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    pixelIQR = Q3-Q1
    arr_std = pixelIQR.to_numpy()
    arr_std.reshape((28,28))
    img = arr_std.reshape((28,28))
    plt.imshow(img, cmap='hot',vmin=0,vmax=255)
    plt.colorbar(label = "IQR")
    plt.xticks([])
    plt.yticks([])
    if filename != "":
        plt.savefig(filename)
    plt.show()

#muestra un ejemplo para cada clase
def mostrar_ejemplos(clase1, clase2):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes[0, i].imshow(X[y == clase1].iloc[i].values.reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f'Clase {clase1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(X[y == clase2].iloc[i].values.reshape(28, 28), cmap='gray')
        axes[1, i].set_title(f'Clase {clase2}')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(f'imagenes/Clase{clase1}_vs_Clase{clase2}.png')
    plt.show()
    

#Para generar los conjuntos de atributos utilizados en el punto 2
def visualizarConjuntosPixeles(conjuntos, nombresConjuntos,filename=""):
    for i in range(len(conjuntos)):
        fila_copia = X.iloc[0].copy()
        fila_copia[:] = 0
        for col in conjuntos[i]:
            fila_copia[col] = 1
        img = fila_copia.to_numpy().reshape((28,28))
        plt.subplot(2, int(len(conjuntos)/2), i + 1)
        plt.imshow(img, cmap="copper")
        plt.title(nombresConjuntos[i])
        plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if(filename != ""):
        plt.savefig(filename)
    plt.show()
    
def entrenarYGuardarMetricas(conjuntos,nombres,filenameExactitudes=""):
    atr = ["nombre_conjunto","exactitud","precision_0","recall_0",
           "precision_8","recall_8"]
    df_metricas = pd.DataFrame(columns=atr)
    for i in range(len(conjuntos)):
        #entreno el modelo KNN con los atributos elegidos
        knn.fit(X_train[conjuntos[i]], y_train)
        pred = knn.predict(X_test[conjuntos[i]])
        print(f"------------------------- Metricas de modelo sobre conjunto {nombres[i]} -------------------------")
        #calculo metricas y creo un dataframe para almacenarlas
        exactitud = round(accuracy_score(y_test, pred),3)
        precision0 = round(precision_score(y_test,pred,pos_label= 0),3)
        recall0 = round(recall_score(y_test,pred,pos_label= 0),3)
        precision8 = round(precision_score(y_test,pred,pos_label= 8),3)
        recall8 = round(recall_score(y_test,pred,pos_label= 8),3)
        df_metricas.loc[i] = [nombres[i],exactitud,precision0,recall0,
                              precision8,recall8]
        #muestra metricas por consola
        for metrica, nombre in zip([exactitud,precision0,recall0,
                                    precision8,recall8],
                                   atr[1:]):
            print(nombre+" = "+str(metrica))
    
    #guardo el archivo con las métricas
    if(filenameExactitudes != ""):
        df_metricas.to_csv(filenameExactitudes,index=False)

#genera la lista de nombres de atributos
def numerosAPixeles(numeros):
    pixeles = []
    for n in numeros:
        pixeles.append('pixel'+str(n))
    return pixeles

def rectangulo(esq_izq,base,altura):
    numeros = []
    for i in range(base):
        for j in range(altura):
            numeros.append(esq_izq+i+28*j)
    return numerosAPixeles(numeros)

def cuadrado(esq_izq,tam_lado):
    return rectangulo(esq_izq,tam_lado,tam_lado)
    


def pixelesDispersos(cant_pixeles, lim_inf,lim_sup):
    np.random.seed(24306470)
    aleatorios = np.random.randint(lim_inf,lim_sup,2*cant_pixeles)
    numeros = []
    for i in range(cant_pixeles):
        numeros.append(aleatorios[i]+28*aleatorios[len(aleatorios)-1-i]-1)
        
    return numerosAPixeles(numeros)
    
#%% Tamaño del dataset, cantidad de clases y etiquetas para cada una
print("Tamaño del dataset:", df.shape)
print("Cantidad de clases:", y.nunique())
print("Clases presentes:", y.unique())

#%% Visualizar una imagen por clase

#Tomo la primera imagen de cada clase 

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"] 
#Nombre de clases obtenidos del repositorio https://github.com/zalandoresearch/fashion-mnist

plt.figure(figsize=(12, 6))
for i in range(10):
    img = X[y == i].iloc[0].values.reshape(28, 28)
    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"{i}: {class_names[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('imagenes/imagen_por_clase.png')
plt.show() 
#%% Distribución de clases
sns.countplot(x=y)
plt.title("Distribución de clases")
plt.xlabel("Etiqueta/Label")
plt.ylabel("Cantidad de imágenes")
plt.savefig('imagenes/distribucion.png')
plt.show() 
#Las clases estan distribuidas por igual

#%% Heatmap de IQR de todas las clases
visualizarIQRPorPixel(X,"IQR-por-pixel")

#%% Comparación de clases

#Selecciono todas las imagenes de la clase i y calcule el valor promedio de cada pixel de esa clase,luego lo transformo en una imagen

plt.figure(figsize=(12, 6))
for i in range(10):
    mean_img = X[y == i].std().to_numpy().reshape(28, 28) 
    plt.subplot(2, 5, i + 1)
    plt.imshow(mean_img, cmap="gray")
    plt.title(f"Clase {i}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('imagenes/imagen_promedio_por_clase.png')
plt.show()


#%% Comparación de IQR entre clases
# 1 contra 2
pantsYSueters = X[(y== 1 ) | (y == 2)]
visualizarIQRPorPixel(pantsYSueters,"IQR-1vs2")
# 2 contra 4 contra 6
suetersCoatsShirts = X[(y== 2 ) | (y == 4) | (y == 6 )]
visualizarIQRPorPixel(suetersCoatsShirts,"IQR-2vs4vs6")

#%% promedio de píxeles por cada clase
plt.suptitle("Imagen promedio por clase")

# Comparar clase 2 vs clase 1
mostrar_ejemplos(2, 1)

# Comparar clase 2 vs clase 6
mostrar_ejemplos(2, 6)

#%% Clase 8 - Variabilidad interna
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
imgs = X[y == 8].sample(10)
for i, ax in enumerate(axes.flat):
    ax.imshow(imgs.iloc[i].values.reshape(28, 28), cmap="gray")
    ax.axis('off')
plt.suptitle("10 ejemplos de clase 8 (Bag)")
plt.savefig('imagenes/clase8_variabilidad_interna.png')
plt.show()
visualizarIQRPorPixel(X[y == 8],"IQR-bolsos")


# --- Clasificación binaria con train-test y knn ---
#%% Ejercicio 2.a - Filtrar clases 0 y 8
df_08 = df[df['label'].isin([0, 8])]
primera_col = df_08.columns[0]
X_08 = df_08.drop(columns=[primera_col,'label'])
y_08 = df_08['label']

visualizarIQRPorPixel(X_08,"IQR-0-8")
print("Analizar balanceo")
print("Clases presentes:", y_08.unique())
print("Cantidad de imágenes por clase:")
print(y_08.value_counts())
sns.countplot(x=y_08)
plt.title("Distribución de clases 0 y 8")
plt.savefig("imagenes/dist_clases_0_8.png")
plt.show()

#%% Ejercicio 2.b - Separar en train y test

X_train, X_test, y_train, y_test = train_test_split(X_08, y_08, test_size=0.2, stratify=y_08, random_state=42)
#Del dataset uso un 80% para entrenar el modelo y un 20% para test
#Tamaño entrenamiento: 11200
#Tamaño test: 2800

#%% Ejercicio 2.c - Probar diferentes combinaciones de 3 atributos

#KNN con 3 vecinos
knn = KNeighborsClassifier(n_neighbors=3)

combinaciones3atr = [
    ['pixel200', 'pixel300', 'pixel400'], #Dispersos en centro superior
    ['pixel100', 'pixel150', 'pixel250'], #Dispersos derecha superior
    ['pixel500', 'pixel600', 'pixel700'], #En Escalera sector inferior
    ['pixel41','pixel69','pixel97'], #Continuos verticales, centro superior
    ['pixel124','pixel125','pixel126'], #Continuos horizontales, centro superior
    ['pixel685','pixel713','pixel741'], #Continuos verticales, centro inferior
    ['pixel656','pixel657','pixel658'], #Continuos horizontales, centro inferior
    ['pixel181','pixel455','pixel467']  #Triangulo en centro
    ]

nombresCombinaciones3atr = ["3A",
                        "3B",
                        "3C",
                        "3D",
                        "3E",
                        "3F",
                        "3G",
                        "3H"]

visualizarConjuntosPixeles(combinaciones3atr, nombresCombinaciones3atr,
                           "pixeles-3-atributos")
entrenarYGuardarMetricas(combinaciones3atr,nombresCombinaciones3atr,
                          "metricas-bin-knn3-3-atributos.csv")

#%% 2c - diferentes combinaciones de diferentes cantidades de atributos
combinacionesVariosAtr = [
    ['pixel0','pixel1','pixel28',
     'pixel26','pixel27','pixel55',
     'pixel728','pixel756','pixel757',
     'pixel755','pixel782','pixel783'],#esquinas 12 px
    pixelesDispersos(12, 4, 24),#dispersos 12px
    rectangulo(286,1,6)+rectangulo(300,1,6),#columnas zonas IQR 12px
    cuadrado(40,3),#cuadrado superior centro 9px
    cuadrado(376,3),#cuadrado centro 9px
    cuadrado(685,3),#cuadrado inferior centro 9px
    pixelesDispersos(6, 4, 24), #dispersos 6px
    numerosAPixeles(range(368,392,4))#fila medio 6px
    
    ] 

nombresCombinacionesVariosAtr = [
    "Esquinas 12px",
    "Dispersos \n12px",
    "Columnas \n12px",
    "C1 9px",
    "C2 9px",
    "C3 9px" , 
    'Dispersos \n6px',
    'Fila medio \n6px'
    ]
visualizarConjuntosPixeles(combinacionesVariosAtr, nombresCombinacionesVariosAtr,"pixeles-n-atributos")
entrenarYGuardarMetricas(combinacionesVariosAtr,
                                         nombresCombinacionesVariosAtr,
                                         "metricas-bin-knn3-n-atributos.csv")

#%% Ejercicio 2.d - Comparar modelos con distinta cantidad de atributos y distintos k
conjuntos = [
    ['pixel41','pixel69','pixel97'],# 3D, continuos verticales superior 3px
    cuadrado(40,3),#C1 cuadrado superior centro 9px
    pixelesDispersos(16, 4, 24), #dispersos 12px
    numerosAPixeles(range(364,392))#fila medio 6px 
    ]

nombres = ["3D 3px","C1 9px","Dispersos 12px",'Fila medio 6px']


k_values = [ 3, 9, 15, 21, 27, 33,39]

dicts_k_metricas= []
for i in range(len(conjuntos)):
    for k in k_values:
        #entreno con el k
        knn_k = KNeighborsClassifier(n_neighbors=k)
        knn_k.fit(X_train[conjuntos[i]], y_train)
        pred_k = knn_k.predict(X_test[conjuntos[i]])
        #métricas del modelo
        exactitud = round(accuracy_score(y_test, pred_k),3)
        precision0 = round(precision_score(y_test,pred_k,pos_label= 0),3)
        recall0 = round(recall_score(y_test,pred_k,pos_label= 0),3)
        precision8 = round(precision_score(y_test,pred_k,pos_label= 8),3)
        recall8 = round(recall_score(y_test,pred_k,pos_label= 8),3)
        f10 = round(f1_score(y_test,pred_k,pos_label= 0),3)
        f18 = round(f1_score(y_test,pred_k,pos_label= 8),3)
        #guardo en una lista de diccionarios para transformarlos en dataframe
        dict_exactitud = {'nombre':nombres[i],'k':k,'exactitud':exactitud,
                          'precision_0':precision0,'recall_0':recall0,
                          'precision_8':precision8,'recall_8':recall8,
                          'f1_0':f10,'f1_8':f18}
        dicts_k_metricas.append(dict_exactitud)
#creo y guardo el dataframe
df_metricas_k = pd.DataFrame(data=dicts_k_metricas)   
df_metricas_k.to_csv("metricas-bin-knn-k-variable.csv",index=False) 


#%% Visualización de valores de metricas para valores distintos de k

#colores para visualización de cada conjunto
colores = ['blue','red','purple','green']

#metricas y nombre para visualizar en graficos de linea
metricas = ['exactitud','precision_0','recall_0','precision_8','recall_8',
            'f1_0','f1_8']
nombres_metricas = ['Exactitud', 'Precisión (clase 0)','Exhaustividad(clase 0)',
                    'Precisión (clase 8)','Exhaustividad(clase 8)', 'F1 (clase 0)','F1 (clase 8)']
for i in range(len(metricas)):
    fig, ax = plt.subplots(figsize=(20,5))
    plt.title(nombres_metricas[i]+' del modelo KNN según valor de k para conjuntos de distintos tamaños',size=16)
    plt.xlabel('Valor de k',size=14)
    plt.ylabel(nombres_metricas[i],size=14)
    plt.ylim(0.85, 0.975)
    plt.xticks(list(range(3, 40,6)))
    ax.grid(True)
    for j in range(len(nombres)):
        #grafico linea para un conjunto
        df_conjunto = df_metricas_k[df_metricas_k["nombre"] == nombres[j]]
        ax.plot('k',metricas[i],data=df_conjunto, marker='o', linestyle='-', 
                color=colores[j],label=nombres[j])
    
    #dejo la leyenda de los colores fija en el centro
    ax.legend(framealpha=0.5,loc='lower center')    
    
    #dibujo y guardo el grafico de la metrica con todos los conjuntos
    plt.tight_layout()
    plt.savefig(metricas[i]+"-knn-k-variable")
    plt.show()
    
#%% --- Clasificación multiclase, k-folding con arboles de decisión ---

# Cargamos los datos

clases = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

y = df['label']
x = df.drop(columns=['label'])  

#%% A) Separamos en conjuntos de desarrollo (80%) y evaluación held-out (20%)

x_dev, x_eval, y_dev, y_eval = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

#%% B) Evaluamos el rendimiento en entrenamiento según profundidad

profundidades = list(range(1, 11))
train_scores = []

for profundidad in profundidades:
    tree = DecisionTreeClassifier(max_depth=profundidad, random_state=42)
    tree.fit(x_dev, y_dev)
    train_scores.append(tree.score(x_dev, y_dev))

plt.figure(figsize=(10, 6))
plt.plot(profundidades, train_scores, 'o-', color='blue', label='Exactitud (Train)')
plt.xlabel('Profundidad del árbol')
plt.ylabel('Exactitud')
plt.title('Rendimiento en entrenamiento vs Profundidad del árbol')
plt.xticks(profundidades)
plt.grid(True)
plt.show()

#%% C) Experimento con validacion cruzada (k-folding)

profundidades = list(range(1, 11))
nsplits = 3
kf = KFold(n_splits=nsplits, shuffle=True, random_state=42)

resultados = np.zeros((nsplits, len(profundidades)))  # filas: folds, columnas: modelos

for i, (train_index, test_index) in enumerate(kf.split(x_dev)):
    kf_x_train, kf_x_test = x_dev.iloc[train_index], x_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]

    for j, hmax in enumerate(profundidades):
        arbol = DecisionTreeClassifier(max_depth=hmax, random_state=42)
        arbol.fit(kf_x_train, kf_y_train)
        pred = arbol.predict(kf_x_test)
        score = accuracy_score(kf_y_test, pred)
        resultados[i, j] = score

# Promedio de scores sobre los folds

scores_promedio = resultados.mean(axis=0)

# Mostramos los resultados promedio por profundidad

for i, d in enumerate(profundidades):
    print(f"Score promedio del modelo con profundidad = {d}: {scores_promedio[i]:.4f}")

#%% D) Entrenamos el mejor modelo en todo el dev y evaluamos en held-out

mejor_profundidad = profundidades[np.argmax(scores_promedio)]
print(f"Profundidad óptima elegida según CV: {mejor_profundidad}")

arbol_elegido = DecisionTreeClassifier(max_depth=mejor_profundidad, random_state=42)
arbol_elegido.fit(x_dev, y_dev)

# Evaluamos en held-out

y_pred = arbol_elegido.predict(x_eval)
heldout_acc = accuracy_score(y_eval, y_pred)
print(f"Exactitud en conjunto held-out: {heldout_acc:.4f}")

# Matriz de confusión

cm = confusion_matrix(y_eval, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
disp.plot(cmap='Blues', values_format='d', xticks_rotation=45)
plt.title('Matriz de confusión sobre el conjunto held-out')
plt.xlabel("Clase predicha", fontsize=12)
plt.ylabel("Clase real", fontsize=12)
plt.show()

