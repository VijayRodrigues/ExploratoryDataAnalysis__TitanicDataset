# ExploratoryDataAnalysis__TitanicDataset


## Description

This is an Exploratory Data Amalysis on the Titanic Dataset where the intention is to handle missing data values. In this analysis missing values are treated accordingly for the columns Age, Cabin and Embarked and understand whether we can come to some kind of conclusion or not.



## Installation

I have used **Google Colaboratory** to code the file. But I have also uploaded **.ipynb file** if Colaboratory isn't accessible.


The following libraries have been used:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

```


## Result

The below is the screenshot for the histogram of "Age" column before treating the missing values

![Titanic_AGe_before](https://user-images.githubusercontent.com/72039550/116579696-45d45600-a930-11eb-8d64-ec9e623703ae.jpg)

---------------------------------------------------------------------------------------------------------------------------------------

Used the following code to **Group** Age based on **Sex** to get their means and replace the missing values using **fillna()** function.

```python
age_df["Age"] = age_df.groupby("Sex").transform(lambda x: x.fillna(x.mean()))
```

The following screenshot shows the histogram post missing values are treated

![Titanic_Age_after](https://user-images.githubusercontent.com/72039550/116580091-abc0dd80-a930-11eb-938f-e078cdb85e17.jpg)

**The histogram shows no much difference in the result even after replacing the missing "age" values with the mean. It goes same if we replaced with median too. Both the above methods add unwantted patterns. We can go ahead with relevant machine learning models to explore further.**
