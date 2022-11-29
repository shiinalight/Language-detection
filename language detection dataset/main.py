import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

data = pd.read_csv("Language Detection.csv")
print(data.head(10))