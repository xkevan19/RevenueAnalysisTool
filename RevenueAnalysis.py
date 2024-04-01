import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV


def predict_new_data(best_model, new_df_without_total_sales):
    # Make predictions on the preprocessed new data using the best model
    new_predictions = best_model.predict(new_df_without_total_sales)

    return new_predictions


def train_model(X_train, y_train):
    # Define the grid of hyperparameters to search over
    param_grid = {
        'fit_intercept': [True, False],
    }

    # Initialize the Linear Regression model
    linear_reg_model = LinearRegression()

    # Initialize GridSearchCV with the model and parameter grid
    grid_search = GridSearchCV(estimator=linear_reg_model, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=3)

    # Perform grid search on the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and the corresponding model
    best_model = grid_search.best_estimator_

    return best_model


def load_and_process_data(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Rename and extract relevant features
    df_copy = df_copy.rename(
        columns={'Gross sales': 'Gross_sales', 'Net sales': 'Net_sales', 'Total sales': 'Total_sales'})
    df_copy['Returns'] = df_copy['Returns']
    df_copy['Coupons'] = df_copy['Coupons']
    df_copy['Taxes'] = df_copy['Taxes']
    df_copy['Shipping'] = df_copy['Shipping']

    # Optionally, drop the original columns if you don't need them anymore
    df_copy = df_copy.drop(columns=['Date'])

    # Step 3: Define the Target Variable
    target_variable = 'Total_sales'

    # Split the data into features (X) and target variable (y)
    X = df_copy.drop(columns=[target_variable])  # Features
    y = df_copy[target_variable]  # Target variable

    # Split the dataset into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Function to detect outliers using IQR (Interquartile Range)
    def detect_outliers_iqr(data, threshold=1.5):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)

    # Detect outliers in the numerical features of the dataset
    outliers = detect_outliers_iqr(df_copy.drop(columns=[target_variable]))

    # Count the number of outliers in each feature
    outliers_count = outliers.sum()

    # Visualize outliers using box plots
    plt.figure(figsize=(12, 6))
    plt.boxplot(df_copy[outliers.columns[outliers.any()]], labels=outliers.columns[outliers.any()].tolist())
    plt.title('Box plot of Numerical Features (Excluding Outliers)')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.show()

    # Handle outliers (e.g., remove them)
    # Assuming we want to remove rows containing outliers
    cleaned_df = df_copy[~outliers.any(axis=1)]

    return cleaned_df, X_train, X_test, y_train, y_test, outliers_count


class RevenueAnalysis:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CSV File Loader")

        self.load_button = ttk.Button(self.root, text="Load CSV", command=self.load_csv)
        self.load_button.pack(pady=10)

        self.status_label = ttk.Label(self.root, text="Select a CSV file to load.")
        self.status_label.pack(pady=5)

        self.predictions_label = ttk.Label(self.root, text="Predictions:")
        self.predictions_label.pack(pady=5)

        self.predictions_list = tk.Listbox(self.root, height=10, width=50)
        self.predictions_list.pack(pady=5)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.status_label.config(text="Loading CSV file...")
            cleaned_df, X_train, X_test, y_train, y_test, outliers_count = load_and_process_data(file_path)
            self.status_label.config(text=f"Outliers detected: {outliers_count}")
            self.status_label.update_idletasks()
            self.status_label.config(text="Training model...")
            best_model = train_model(X_train, y_train)

            # Load the new data CSV file into a DataFrame (assuming named 'new_data.csv')
            new_df = pd.read_csv('new_data.csv')

            # Preprocess the new data to match the column names used during training
            new_df_copy = new_df.copy()
            new_df_copy = new_df_copy.rename(
                columns={'Gross sales': 'Gross_sales', 'Net sales': 'Net_sales', 'Total sales': 'Total_sales'})
            new_df_copy['Returns'] = new_df_copy['Returns']
            new_df_copy['Coupons'] = new_df_copy['Coupons']
            new_df_copy['Taxes'] = new_df_copy['Taxes']
            new_df_copy['Shipping'] = new_df_copy['Shipping']

            # Drop any additional columns not present during training
            new_df_copy = new_df_copy.drop(columns=['Date'])

            # Drop the 'Total_sales' column from the new data
            new_df_without_total_sales = new_df_copy.drop(columns=['Total_sales'])

            # Load the actual revenue from new data
            actual_revenue = new_df_copy['Total_sales']

            new_predictions = predict_new_data(best_model, new_df_without_total_sales)

            # Display the predictions
            self.status_label.config(text="Predictions on new data:")
            self.predictions_list.delete(0, tk.END)  # Clear previous predictions
            for i, prediction in enumerate(new_predictions):
                self.predictions_list.insert(tk.END, f"Prediction {i + 1}: {prediction:.2f}")

            # Show the predictions listbox
            self.predictions_list.pack()

            # Visualize the comparison between predicted and actual revenue
            plt.figure(figsize=(12, 6))

            # Scatter plot
            plt.subplot(1, 2, 1)
            plt.scatter(actual_revenue, new_predictions)
            plt.xlabel('Actual Revenue')
            plt.ylabel('Predicted Revenue')
            plt.title('Actual vs. Predicted Revenue')

            # Histogram of errors
            errors = actual_revenue - new_predictions
            plt.subplot(1, 2, 2)
            plt.hist(errors, bins=20, edgecolor='black')
            plt.xlabel('Errors')
            plt.ylabel('Frequency')
            plt.title('Distribution of Errors')

            plt.tight_layout()
            plt.show()

            # Line chart for trend over time (if applicable)
            # Assuming the new data includes a timestamp column
            if 'Date' in new_df_copy.columns:
                plt.figure(figsize=(12, 6))
                plt.plot(new_df_copy['Date'], actual_revenue, label='Actual Revenue', marker='o')
                plt.plot(new_df_copy['Date'], new_predictions, label='Predicted Revenue', marker='x')
                plt.xlabel('Date')
                plt.ylabel('Revenue')
                plt.title('Actual vs. Predicted Revenue Over Time')
                plt.xticks(rotation=45)
                plt.legend()
                plt.tight_layout()
                plt.show()

    def main(self):
        self.root.mainloop()


if __name__ == "__main__":
    revenue_analysis = RevenueAnalysis()
    revenue_analysis.main()
