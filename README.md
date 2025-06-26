# **ML_Speed_Dating_Predictions**

Repository for a ML project in which I will try to predict whether a speed date between two given individuals will be successful or not.

### **Machine Learning Problem**

Speed Dating is a social activity in which participans have short "dates" or conversations with other participants in order to get to know them and decide whether they feel attracted to, or interested in them. If both participants in a date like the person in front of them, we found what we are looking for: a "match".

The aim of this project is to train a model to predict whether two individuals in a speed date will match or not. Additionally, I will examine what features are relevant to predict a match, so that dates can be arranged to maximize matches. Thus, participants will be happier with the outcome of other speed dating events and will continue to attend or will be more likely to recommend them.

### **Data**

##### Source and download

The data used for this project was obtained from OpenML, the dataset is public and can be accessed at: [Speed Dating Dataset - OpenML](https://www.openml.org/search?type=data&sort=runs&status=active&id=40536) or in this repository, in the data_sample directory:
```python
!pip install scipy
from scipy.io import arff
data, meta = arff.loadarff("./src/data_sample/speed_dating.arff")
```
Useful information on the variables can also be found in the same directory, file: ´feature_information.txt´

##### Description

The dataset contains data from several experimental speed dating events held by Columbia Business School. These data was originally gathered to study gender differences in mate selection. However, it contains a number of variables with information about both individuals taking part in each date, which is very useful to train a machine learning model.   

The data consists of information gathered from participants on several topics, such as what they find important in a potential partner or their hobbies. At the end of the date, they are asked if they would like to see the other person again, if both say yes: it's a match! Data is labelled 