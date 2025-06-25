# **ML_Speed_Dating_Predictions**

Repository for a ML project in which I will try to predict whether a speed date between two given individuals will be successful or not.

### **Machine Learning Problem**

Speed Dating is a social activity in which participans have short "dates" or conversations with other participants in order to get to know them and decide whether they feel attracted to, or interested in, them. If both participants in a date like the person in front of them, there is a "match".

The aim of this project is to train a model to predict whether two individuals speed dating will match or not. Additionally, I will examine what features are relevant to predict a match, so that dates can be arranged to maximize matches. Thus, participants will be happier with the outcome of other speed dating events and will continue to attend or recommend them.

### **Data**

The data used for this project were obtained from OpenML, the dataset is public and can be accessed at: [Speed Dating Dataset](https://www.openml.org/search?type=data&sort=runs&status=active&id=40536) or in this repository, in the data_sample directory:
```python
data, meta = arff.loadarff("./src/data_sample/speed_dating.arff")
```
Useful information on the variables can also be found in the same directory, in ´feature_information.txt´
They contain data from several experimental speed dating events held by Columbia Business School. These data were originally gathered to determine gender differences in mate selection and contain a number of variables with information about both individuals taking part in each date. The data are also labelled (match [0 or 1]), what makes them suitable to solve a supervised classification problem.