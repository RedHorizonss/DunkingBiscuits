import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import NearestNeighbors

from scipy.signal import find_peaks

sns.set_theme(style="whitegrid", font_scale=1.25 ,rc={"xtick.bottom" : True, "ytick.left" : True, 
                                    "axes.spines.bottom" : True, "axes.spines.left" : True,
                                    "axes.spines.right" : False, "axes.spines.top" : False,
                                    'axes.linewidth': 2, 'axes.edgecolor':'black',
                                    'axes.formatter.limits': (-2, 2), 'axes.formatter.use_mathtext' : True})

palette = ["#21918c", "#addc30", "#472d7b", "#5ec962", "#3b528b", "#28ae80"]
sns.set_palette(palette)

class biscuit_models():
    def __init__(self, df):
        self.df = df
            
    def define_features(self, feature_columns, target_column):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.X = self.df[self.feature_columns]
        self.y = self.df[self.target_column]
        
    def polynommial_features(self, degree=2):
        poly = PolynomialFeatures(degree=degree)
        self.X = poly.fit_transform(self.X)
        
    def train_test_split_data(self, random_state=42, test_size=0.2, stratify=None):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                                test_size=test_size, random_state=random_state, stratify=stratify)
    
    def train_model(self, model_type='linear_regression', **kwargs):
        if model_type == 'linear_regression':
            self.model = LinearRegression(**kwargs)
        elif model_type == 'random_forest_regression':
            self.model = RandomForestRegressor(**kwargs)
        elif model_type == 'random_forest_classifier':
            self.model = RandomForestClassifier(**kwargs)
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
        
    def plot_predictions(self, xlabel='Actual', ylabel='Predicted', save_image=False, filename=None):
        plt.figure(figsize=(6, 5))
        plt.scatter(self.y_test, self.y_pred, label=f'$R^{2}$: {self.r2:.3f}', alpha=0.8)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()],
                 color='black', lw=2, linestyle='--', label='y = x')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        if save_image:
            plt.savefig(filename, dpi = 900)
        plt.show()
        
    def plot_residuals(self, save_image=False, filename=None):
        residuals = self.y_test - self.y_pred
        plt.figure(figsize=(6, 5))
        plt.scatter(self.y_pred, residuals, alpha=0.8)
        plt.axhline(y=0, color='black', lw=2, linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Prediction Residuals')
        if save_image:
            plt.savefig(filename, dpi = 900)
        
    def plot_feature_importances(self, save_image=False, filename=None):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(6, 5))
        container = plt.bar(range(self.X_train.shape[1]), importances[indices])
        plt.bar_label(container, labels=[f'{importances[i]:.3f}' for i in indices], label_type='center', color='white')
        plt.xticks(range(self.X_train.shape[1]), [self.feature_columns[i] for i in indices], rotation=90)
        plt.title('Feature Importances')
        if save_image:
            plt.savefig(filename, dpi = 900)
        plt.show()


def washburn_eqn(df):
    df['L_squared'] = df['L']**2
    df['cos(phi)'] = np.cos(df['phi'])
    df['1/cos(phi)'] = 1/df['cos(phi)']
    
    df['1/gamma'] = 1/df['gamma']
    df['1/t'] = 1/df['t']
    
    df['washburn_eqn'] = df['L_squared'] * 2*df['eta'] * df['1/gamma'] *df['1/t']*df['1/cos(phi)']
    
    return df

def scatter_plot_3D(x, y, z, color_by ,x_label, y_label, z_label, eye_x, eye_y, eye_z):
    fig = go.Figure(data=[go.Scatter3d(
        x = x,
        y = y,
        z = z,
        mode='markers',
        marker=dict(
            size=5,
            color=color_by,
            colorscale='Viridis',   
            opacity=0.8,
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title=x_label, title_font=dict(size=20)),
            yaxis=dict(title=y_label, title_font=dict(size=20)),
            zaxis=dict(title=z_label, title_font=dict(size=20)),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=500,
        width=500,
        template='plotly_white',
        font=dict(family="Helvetica",
                size=14,
                color="black",),
        
        scene_camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=eye_x, y=eye_y, z=eye_z))
        )

    fig.show()
    
def get_peaks(datapoints, bw, tolerance = 1e6):
    density = sns.kdeplot(datapoints, bw_adjust=bw).get_lines()[-1].get_data()
    peaks, _ = find_peaks(density[1], height=tolerance)
    plt.close()
    return density[0][peaks], density[1][peaks]

def follow_peak_evolution(datapoints, bw_values, tolerance):
    mode_evolution = []
    for i in bw_values:
        x_peaks, y_peaks = get_peaks(datapoints, i, tolerance)
        
        for value in x_peaks:
            mode_evolution.append([value, i])
    
    plt.close()
    mode_evolution = np.array(mode_evolution)
    return mode_evolution

def trace_modes(datapoints, bw_values, tolerance, precentage_range, number_of_modes):
    mode_value = 1
    mode_evolution = []
    trace_modes = {}

    upper_percent = 1 + precentage_range
    lower_percent = 1 - precentage_range

    for i in bw_values:
        density = sns.kdeplot(datapoints, bw_adjust=i).get_lines()[-1].get_data()
        
        peaks, _ = find_peaks(density[1], height=tolerance)  # density[1] contains the y-values of the KDE
        
        # Stores all possible modes found in the KDE
        for value in peaks:
            mode_evolution.append([density[0][value], i])
            
        # If no modes have been found yet, add all peaks to the dictionary (these are the first modes found in the KDE)
        if trace_modes == {}:
            for value in peaks:
                trace_modes[f'Mode {mode_value}'] = [[density[0][value],i]]
                # Mode value keeps track of the number of modes found
                mode_value += 1
        # If modes have been found, add new peaks to the dictionary if they are close to the previous peaks
        else:
            # peaks just added lets us know which peaks have been added to the dictionary so if the number of modes are not reached, we can make a new mode without repeating peaks
            peaks_just_added = []
            # First we add the peaks that are close to the previous peaks
            for key in trace_modes.keys():
                for item in density[0][peaks]:
                    # The peaks are added if they are within 0.5% of the previous peak
                    if (item < trace_modes[key][-1][0]*upper_percent) and (item > trace_modes[key][-1][0]*lower_percent):
                        trace_modes[key].append([item,i])
                        peaks_just_added.append(item)
                            
            # If the number of modes have not been reached, add new modes with the remaining peaks               
            if mode_value <= number_of_modes:
                if len(np.setdiff1d(density[0][peaks], peaks_just_added)) > 0:
                    for item in np.setdiff1d(density[0][peaks], peaks_just_added):
                        trace_modes[f'Mode {mode_value}'] = [[item,i]]
                        mode_value += 1
            
    plt.close()

    mode_evolution = np.array(mode_evolution)
    
    for key in trace_modes:
        trace_modes[key] = np.array(trace_modes[key])
    
    return trace_modes, mode_evolution

def trace_centers(trace_modes):
    centroids = []
    for key in trace_modes.keys():
        center = np.mean(trace_modes[key], axis=0)
        centroids.append(center)
    
    return np.array(centroids)