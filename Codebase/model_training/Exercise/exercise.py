import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
df1 = pd.read_csv('../../../Data/exercise/exercises.csv')
df1.head()