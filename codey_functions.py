import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

sns.set_theme(style="whitegrid", font_scale=1.25 ,rc={"xtick.bottom" : True, "ytick.left" : True, 
                                    "axes.spines.bottom" : True, "axes.spines.left" : True,
                                    "axes.spines.right" : False, "axes.spines.top" : False,
                                    'axes.linewidth': 2, 'axes.edgecolor':'black'})

palette = ["#21918c", "#5ec962", "#3b528b", "#440154", "#fde725"]
sns.set_palette(palette)

class biscuit_models():
    def __init__(self, df, washburn_eqn=False):
        self.df = df
        
        if washburn_eqn:
            self.washburn_eqn()
            
    def define_features(self, feature_columns, target_column):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.X = self.df[self.feature_columns]
        self.y = self.df[self.target_column]
        
    def washburn_eqn(self):
        self.df['L_squared'] = self.df['L']**2
        self.df['cos(phi)'] = np.cos(self.df['phi'])
        self.df['2eta'] = 2*self.df['eta']
        self.df['1/gamma'] = 1/self.df['gamma']
        self.df['1/t'] = 1/self.df['t']
        self.df['1/cos(phi)'] = 1/self.df['cos(phi)']
        self.df['washburn_eqn'] = self.df['L_squared'] * self.df['2eta'] * self.df['1/gamma'] *self.df['1/t']*self.df['1/cos(phi)']
        
    def polynommial_features(self, degree=2):
        poly = PolynomialFeatures(degree=degree)
        self.X = poly.fit_transform(self.X)
        
    def train_test_split_data(self, random_state=42, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                                test_size=test_size, random_state=random_state)
    
    def train_model(self, model_type='linear_regression', **kwargs):
        if model_type == 'linear_regression':
            self.model = LinearRegression(**kwargs)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(**kwargs)
        else:
            raise ValueError('Invalid model_type. Must be either linear_regression or random_forest')
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate_model(self):
        self.y_pred = self.model.predict(self.X_test)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.mae = mean_absolute_error(self.y_test, self.y_pred)
        self.r2 = r2_score(self.y_test, self.y_pred)
        
    def print_metrics(self):
        print(f'Mean Squared Error: {self.mse:.3e}')
        print(f'Mean Absolute Error: {self.mae:.3e}')
        print(f'Root Mean Squared Error: {np.sqrt(self.mse):.3e}')
        print(f'$R^{2}$: {self.r2:.3f}')
        
    def plot_predictions(self, xlabel='Actual', ylabel='Predicted'):
        plt.figure(figsize=(6, 5))
        plt.scatter(self.y_test, self.y_pred, label=f'$R^{2}$: {self.r2:.3f}', alpha=0.8)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()],
                 color='black', lw=2, linestyle='--', label='y = x')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
        
    def plot_residuals(self):
        residuals = self.y_test - self.y_pred
        plt.figure(figsize=(6, 5))
        plt.scatter(self.y_pred, residuals, alpha=0.8)
        plt.axhline(y=0, color='black', lw=2, linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Prediction Residuals')
        
    def plot_feature_importances(self):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(6, 5))
        container = plt.bar(range(self.X_train.shape[1]), importances[indices])
        plt.bar_label(container, labels=[f'{importances[i]:.3f}' for i in indices], label_type='center', color='white')
        plt.xticks(range(self.X_train.shape[1]), [self.feature_columns[i] for i in indices], rotation=90)
        plt.title('Feature Importances')
        plt.show()
