## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
Developed  By: Preethi A A
Register Number : 212222110035
```
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/7fc9b93c-2a8b-4253-b2f9-124e7bd48c43)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/b8b0ce0d-9ed1-4764-b2d4-419051c9588d)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-10-05 141305](https://github.com/user-attachments/assets/cf23c9b2-7ef7-4b75-9e3a-d6bc82dcc7df)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-10-05 141438](https://github.com/user-attachments/assets/6da3a60a-ebe3-4914-a5f0-9db7823df6ec)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![Screenshot 2024-10-05 141532](https://github.com/user-attachments/assets/101fd240-6408-4593-a239-6bdfda376049)

```
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2024-10-05 141720](https://github.com/user-attachments/assets/d9b5c1f6-ccbc-48b8-810b-9ce492a7523e)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2024-10-05 141800](https://github.com/user-attachments/assets/03b93f16-02a1-4fd8-a3df-a6449be947ff)

```
pip install --upgrade category_encoders
```
![Screenshot 2024-10-05 141928](https://github.com/user-attachments/assets/d2a220d2-2280-4cce-aedb-e8953dca91ac)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![Screenshot 2024-10-05 142014](https://github.com/user-attachments/assets/d8433539-53b7-4f95-934a-c365cedd0b73)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![Screenshot 2024-10-05 142241](https://github.com/user-attachments/assets/985487ce-fdba-40ca-81c9-f23fb84ef748)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![Screenshot 2024-10-05 142330](https://github.com/user-attachments/assets/d332faeb-6898-4b65-a797-36dd105e9264)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![Screenshot 2024-10-05 142427](https://github.com/user-attachments/assets/aec8ae6a-5ff5-4d57-9dfe-686847329b0c)

```
df.skew()
```
![Screenshot 2024-10-05 142522](https://github.com/user-attachments/assets/9335b119-d14c-4314-9397-efa0b2ac4b96)

```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2024-10-05 142827](https://github.com/user-attachments/assets/e779b5d8-2dad-4c2e-b38b-4b91dfb03045)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/8b2d2b72-e84d-4629-933c-7fe8ee1f3253)

```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2024-10-05 142953](https://github.com/user-attachments/assets/8781f3c4-ac9a-4fed-83d1-6acc76e5bfe2)

```
np.square(df["Highly Positive Skew"])
```
![Screenshot 2024-10-05 143058](https://github.com/user-attachments/assets/d8f4d56d-d7dd-4b83-839b-5be691cb7d6f)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2024-10-05 143142](https://github.com/user-attachments/assets/e83c885f-ae7f-45cf-a29a-4590e344777a)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
![Screenshot 2024-10-05 143234](https://github.com/user-attachments/assets/91285396-131c-4b68-8cae-68e32bea1f57)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-10-05 143311](https://github.com/user-attachments/assets/adfa37b2-1130-41cd-9847-34bbc3425f62)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```
![Screenshot 2024-10-05 143349](https://github.com/user-attachments/assets/059977a7-0c45-4098-95c4-7a03ad8d67a4)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-10-05 143444](https://github.com/user-attachments/assets/2b7f46bc-7978-477c-843d-ee5102914d04)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-10-05 143524](https://github.com/user-attachments/assets/4c7309e7-efec-40bb-aac7-d7d23daddb8a)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![Screenshot 2024-10-05 143610](https://github.com/user-attachments/assets/cc199c03-0fad-4991-b5b1-08f9243ec79e)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2024-10-05 143646](https://github.com/user-attachments/assets/165fcd69-fd55-46e1-9877-480bba1c49fa)

# RESULT:
       # INCLUDE YOUR RESULT HERE

       
