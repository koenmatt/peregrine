# Important Note:
I have included most related code in 

#### Installing and importing the dependencies


```python
!pip install -r requirements.txt --quiet
```


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```

# Prompt

# EDA (Exploratory Data Analysis)

<!-- "Victorious warriors win first and then go to war, while defeated warriors go to war first and then seek to win." - Sun Tzu, author of the Art of War. 
 -->
Sun Tzu, chinese philospher and military strategist, said it best with his timeless advice: 

**"Know thy self, know thy enemy. A thousand battles, a thousand victories."**

Applying this to our task, 

<!-- The COMMANDER stands for the virtues of wisdom, sincerity, benevolence, courage and strictness.
 -->



```python
FILE_PATH = 'test.csv'
df = pd.read_csv(FILE_PATH)
print("--------------------")
print(f"Shape: {df.shape}")
print("--------------------")
df.head()
```

    --------------------
    Shape: (1459, 80)
    --------------------





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>120</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>60</td>
      <td>RL</td>
      <td>78.0</td>
      <td>9978</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>120</td>
      <td>RL</td>
      <td>43.0</td>
      <td>5005</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>...</td>
      <td>144</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>




```python
print(list(df.columns))
```

    ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']



```python
missing_data = describe_missing_data(df)
missing_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>1456</td>
      <td>0.997944</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>1408</td>
      <td>0.965045</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>1352</td>
      <td>0.926662</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>1169</td>
      <td>0.801234</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>894</td>
      <td>0.612748</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>730</td>
      <td>0.500343</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>227</td>
      <td>0.155586</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>78</td>
      <td>0.053461</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>78</td>
      <td>0.053461</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>78</td>
      <td>0.053461</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_multiple_continuous_distributions(['LotFrontage', 'LotArea', 'MasVnrArea'], df)
```


    
![png](output_10_0.png)
    


# Helper Functions - EXECUTE THIS CELL FIRST


```python
def describe_missing_data(df):
    """Describes the null values present in our dataset"""
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

def plot_multiple_continuous_distributions(col_names, df):
    """Plot the distributions of the list of column names 
    from the dataframe, df."""
    
    num_cols = min(3, len(col_names))
    num_rows = (len(col_names) + 2) // 3  
    
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 4 * num_rows))
    axes = axes.flatten()
    
    for i, name in enumerate(col_names):
        sns.distplot(df[name], ax=axes[i])
        axes[i].set_title(name)
    
    for j in range(len(col_names), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
    
def analyze_distribution(col_name, df): 
    ...
    
def remove_outliers_IQR():
    """Remove outliers using IQR from the dataset"""
    ...

def cluster_graph():
    ...

def cluster_classifier():
    ...

# sns.pairplot(df)

```


```python

```
