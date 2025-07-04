# **ML_Speed_Dating_Predictions**

## Versión en español
\* See English version below

Repositorio para un proyecto de ML en el que entrenaré un modelo de machine learning para predecir si una cita rápida entre dos personas determinadas será exitosa o no.

### **Problema de negocio**

Las citas rápidas (*speed dating*) son una actividad social en la que los participantes tienen «citas» o conversaciones cortas con otros participantes para conocerlos y decidir si se sienten atraídos o interesados por ellos. Si a los dos participantes de una cita les gusta la persona que tienen delante, hemos encontrado lo que buscábamos: un «match».

Para que un evento de citas rápidas se organice y se lleve a cabo correctamente, es recomendable utilizar datos de otros eventos para predecir el resultado de una posible cita y, además, poder encontrar la pareja adecuada para cada persona sabiendo qué aspectos son clave para averiguar si dos personas se van a gustar.

### **Problema de machine learning**

El objetivo de este proyecto es entrenar un modelo para predecir si dos individuos en una cita rápida harán match o no. Además, examinaré qué características son relevantes para predecir una coincidencia, de modo que las citas puedan organizarse para maximizar los matches. Así, los participantes estarán más contentos con el resultado de los nuevos eventos de speed dating y seguirán asistiendo o será más probable que los recomienden.

### **Datos**

##### Fuente y descarga

Los datos utilizados para este proyecto se obtuvieron de OpenML, el dataset es público y se puede acceder a él en: [Speed Dating Dataset - OpenML](https://www.openml.org/search?type=data&sort=runs&status=active&id=40536) o en este repositorio, en el directorio data_sample:
```python
!pip install scipy
from scipy.io import arff
data, meta = arff.loadarff("./src/data_sample/speed_dating.arff")
```
También se puede encontrar información útil sobre las variables en el mismo directorio, archivo: 'feature_information.txt'

##### Descripción

El conjunto de datos contiene datos de varios eventos experimentales de citas rápidas celebrados por la Columbia Business School. Los datos consisten en información de los participantes relativa a varios temas, como lo que consideran importante en una pareja potencial o sus aficiones. Al final de la cita, se les pregunta si les gustaría volver a ver a la otra persona, si ambos dicen que sí: ¡es un match!

### **Solución**

##### Baseline

El primer paso de este proyecto consistió en ajustar 4 modelos de referencia para poder comparar mis resultados y evaluar los progresos realizados. Para ello, elegí un clasificador de árbol de decisión, un clasificador de random forest, un clasificador XGBoost y otro de LightGBM.

Resultados de referencia:

|model| balanced_accuracy|
|--|--|
|LightGBM|0.616981|
|Random Forest| 0.599860|
|Decision Tree| 0.583740|
|XGBoost| 0.563692|

##### Selección preliminar de features

Los siguiente que he hecho ha sido realizar una selección automatizada de features, con el fin de reducir mi número de features (inicialmente 88) para realizar un EDA. Hice 4 selecciones por distintos medios y las probé para comprobar cuál funcionaba mejor. Finalmente elegí una selección de modelo realizada mediante un bosque aleatorio, que constaba de 30 features.

##### EDA

Realicé tanto un análisis univariante como un análisis bivariante de todas mis features con respecto a mi target. No pude reducir más la dimensionalidad en este paso, debido a la gran diferencia entre las frecuencias de ambas clases. Sin embargo, me ayudó a entender mis datos a fondo y descubrir transformaciones que contribuirían a un mejor funcionamiento de mi modelo.

##### Selección final

Una vez creadas las nuevas features que se me habían ocurrido durante los pasos anteriores, así como transformadas las features numéricas para dotarlas de una distribución más gaussiana y escaladas (estos dos procesos son especialmente útiles para los modelos basados en la distancia), realicé una selección final de features del mismo modo que lo había hecho en la selección preliminar. Los mejores resultados fueron los siguientes

|model| features_set |balanced_accuracy|
|--|----|
|XGBoost |RFECV selection| 0.768713|
|LightGBM |RFECV selection| 0.691811|
|KNN |RFECV selection| 0.681842|

Las features seleccionadas mediante RFECV incluyen, en orden descendente de importancia:
* Correlación de intereses: correlación de los intereses o aficiones de ambos participantes, rango: -1, 1
* Correlación de preferencias: correlación de la importancia dada por ambos participantes a determinados atributos a la hora de elegir pareja, rango: -1, 1
* Diferencia en la importancia del atractivo físico: diferencia entre la valoración dada por ambos participantes a la importancia del atractivo, rango: 0, 100
* Diferencia de edad como porcentaje de la edad del segundo participante, rango: 0-100

##### Optimización de hiperparámetros

Teniendo en cuenta los resultados anteriores, decidí optimizar los tres algoritmos anteriores utilizando la biblioteca Optuna, en sus propias palabras "Un marco de optimización de hiperparámetros de código abierto para automatizar la búsqueda de hiperparámetros". Así es como llegué a mi modelo final: LightGBM - 82% de precisión equilibrada en CV.

##### Resultados

Cuando evalué mi modelo con el conjunto de datos de prueba, el resultado fue incluso mejor que con la validación cruzada. El informe de clasificación fue el siguiente:  

|-| class |precision | recall| f1-score | support|
|-|-|-|-|-|-|-|
|-|0| 0.95 | 0.98 | 0.96 | |
|-|1 | 0.88 | 0.74 | 0.80 | 287|
|accuracy|-| - | - | 0. 94 | 1676|
|macro avg|-| 0.91 | 0.86 | 0.88 | 1676|
|weighted avg |- | 0.93 | 0.94 | 0.93 | 1676|
\*Balanced accuracy = macro avg x recall

### ***Estructura del repositorio***

```python
├── src/                # Contiene todos los demás directorios
    ├── data_sample/    # Datos y archivos relacionados
    ├── models/         # Modelo final optimizado
    ├── notebooks/      # Otros notebooks utilizados para probar el código durante el proceso de creación del modelo
    ├── utils/          # Todos los módulos auxiliares utilizados en el proyecto
├── main. ipynb         # Notebook final que contiene todos los pasos seguidos a lo largo del proceso
├── presentacion.pdf    # Documento resumen del proyecto, utilizado como soporte para la presentación
├── README.md           # El archivo que estás leyendo en este momento :)
```

## English Version

Repository for a ML project in which I will train a machine learning model to predict whether a speed date between two given individuals will be successful or not.

### **Business Problem**

Speed Dating is a social activity in which participans have short "dates" or conversations with other participants in order to get to know them and decide whether they feel attracted to, or interested in them. If both participants in a date like the person in front of them, we found what we are looking for: a "match".   

In order for a speed dating event to be properly organised and conducted, it is advisable to use data from other events to predict the outcome of a potential date and, what is more, to be able to find the proper partner for every person kwnowing which aspects are key to figure out if two people will like each other.

### **Machine Learning Problem**

The aim of this project is to train a model to predict whether two individuals in a speed date will match or not. Additionally, I will examine what features are relevant to predict a match, so that dates can be arranged to maximize matches. Thus, participants will be happier with the outcome of new speed dating events and will continue to attend or will be more likely to recommend them.

### **Data**

##### Source and download

The data used for this project was obtained from OpenML, the dataset is public and can be accessed in: [Speed Dating Dataset - OpenML](https://www.openml.org/search?type=data&sort=runs&status=active&id=40536) or in this repository, in the data_sample directory:
```python
!pip install scipy
from scipy.io import arff
data, meta = arff.loadarff("./src/data_sample/speed_dating.arff")
```
Useful information on the variables can also be found in the same directory, file: ´feature_information.txt´

##### Description

The dataset contains data from several experimental speed dating events held by Columbia Business School. These data consists of information from participants relating to several topics, such as what they find important in a potential partner or their hobbies. At the end of the date, they are asked if they would like to see the other person again, if both say yes: it's a match!

### **Solution**

##### Baseline

The first step of this project was fitting 4 baseline models to be able to compare my results and evaluate the progress made. For this purpose, I chose a decision tree classifier, a random forest classifier, an XGBoost classifier and a LightGBM classifier.

Baseline results:

|model|	balanced_accuracy|
|--|--|
|LightGBM|0.616981|
|Random Forest|	0.599860|
|Decision Tree|	0.583740|
|XGBoost|	0.563692|

##### Preliminary feature selection

After that, I conducted an automated feaure selection, in order to reduce my number of features (initially 88) to perform an EDA. I made 4 selections by different means and tried them to check which one worked better. I finally picked up a model selection made by means of a random forest, which comprised 30 features.

##### EDA

I performed both an univariate analysis and a bivariate analysis of all my features against my target. I was not able to further reduce dimensionality in this step, due to the wide gap between frquencies of both classes. Nevertheless, I was able to understand my data thoroughly and to figure out transformations that would help my model perform better.

##### Final selection

After I had created the new features I had come up with during the previous steps, as well as transformed numerical features in order to provide them with a more Gaussian-like distribution and scaled them (these two processes are particularly useful for distance-based models), I made a final feature selection the same way I had done it in the preliminary selection. The best results were as follows:

|model|	features_set	|balanced_accuracy|
|--|--|--|
|XGBoost	|RFECV selection|	0.768713|
|LightGBM	|RFECV selection|	0.691811|
|KNN	|RFECV selection|	0.681842|

RFECV features include, in descending order of importance:
* Interests correlate: correlation of the interests or hobbies of both participants, range: -1, 1
* Preferences correlate: correlation of the importance given by both participants to certain attributes when it comes to choosing a partner, range: -1, 1
* Diference in important of attractiveness: difference between the rating given by both participants to attractiveness importance, range: 0, 100
* Diference of age as percentage of the second participant's age, range: 0-100

##### Hyperparameter optimization

Based on the above results, I decided to fine-tune the three algorithms above using the library Optuna, in their own words "An open source hyperparameter optimization framework to automate hyperparameter search". This is how I got to my final model: LightGBM - 82% balanced accuracy in CV.

##### Results

When I evaluated my model against the test dataset, the result was even better than with cross validation. The classification report was as follows:   

|-| class |precision  |  recall|  f1-score |  support|
|-|-|-|-|-|-|
|-|0|       0.95 |     0.98 |     0.96 |     1389|
|-|1   |    0.88   |   0.74   |   0.80   |    287|
|accuracy|-| - | - |    0.94 |     1676|
|macro avg|-|     0.91 |     0.86    |  0.88     | 1676|
|weighted avg  |- |    0.93   |   0.94  |    0.93    |  1676|
\*Balanced accuracy = macro avg x recall

### ***Repository structure***

```python
├── src/                # It contains all other directories
    ├── data_sample/    # Data and related files
    ├── models/         # Final optimized model
    ├── notebooks/      # Other notebooks used to test code during the modelling process
    ├── utils/          # All auxiliary modules used in the project
├── main.ipynb          # Final notebook containing all steps followed troughout the process
├── presentacion.pdf    # Document summing up the project, used as presentation support
├── README.md           # This file you are reading right now :)
```
