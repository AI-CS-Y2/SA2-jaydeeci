import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class SportsCarPricePredictionSVM:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.df = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_clean_data(self):
        # Load data
        self.df = pd.read_csv(self.data_path)
        print("Initial Data:")
        print(self.df.head())

        # Data Cleaning
        # Drop rows with missing values
        self.df.dropna(inplace=True)

        # Drop duplicate rows
        self.df.drop_duplicates(inplace=True)

        # Clean 'Price (in USD)' column (remove commas)
        self.df['Price (in USD)'] = self.df['Price (in USD)'].apply(lambda x: x.replace(',', '') if isinstance(x, str) else x)

        # Clean other columns
        def replace_string(col, str1, str2):
            self.df[col] = self.df[col].apply(lambda x: str(x).replace(str1, '').replace(str2, '') if isinstance(x, str) else x)

        replace_string('Horsepower', ',', '+')
        replace_string('Torque (lb-ft)', '-', '+')
        replace_string('Torque (lb-ft)', ',', '')
        replace_string('0-60 MPH Time (seconds)', '<', '+')

        # Convert columns to numeric values
        self.df.iloc[:, 3:8] = self.df.iloc[:, 3:8].apply(pd.to_numeric, errors='coerce')

        # Handle missing values in specific columns by replacing with mean values
        self.df['Engine Size (L)'] = self.df['Engine Size (L)'].fillna(self.df['Engine Size (L)'].mean())
        self.df['Torque (lb-ft)'] = self.df['Torque (lb-ft)'].fillna(self.df['Torque (lb-ft)'].mean())

        # Convert columns to integers
        self.df['Engine Size (L)'] = self.df['Engine Size (L)'].astype(int)
        self.df['Horsepower'] = self.df['Horsepower'].astype(int)
        self.df['Torque (lb-ft)'] = self.df['Torque (lb-ft)'].astype(int)
        self.df['0-60 MPH Time (seconds)'] = self.df['0-60 MPH Time (seconds)'].astype(int)
        self.df['Price (in USD)'] = self.df['Price (in USD)'].astype(int)

        # Show cleaned data info
        print("\nData Info after cleaning:")
        print(self.df.info())

    def prepare_data(self):
        # Features and target
        X = self.df[['Year', 'Engine Size (L)', 'Horsepower', 'Torque (lb-ft)', '0-60 MPH Time (seconds)']]  # Features
        y = self.df['Price (in USD)']  # Target variable

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train_model(self):
        # Train the Support Vector Machine (SVM) Regressor model
        self.model = SVR(kernel='rbf')  # Using the radial basis function (RBF) kernel
        self.model.fit(self.X_train_scaled, self.y_train)

    def evaluate_model(self):
        # Make predictions on the test data
        y_pred = self.model.predict(self.X_test_scaled)

        # Evaluate the model
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"\nModel Evaluation:")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R^2 Score: {r2}")

        # Plot the actual vs predicted prices
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, color='blue', alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], color='red', linewidth=2)
        plt.title('Actual vs Predicted Prices (SVM)')
        plt.xlabel('Actual Prices (in USD)')
        plt.ylabel('Predicted Prices (in USD)')
        plt.grid(True)
        plt.show()

    def plot_feature_importance(self):
        # Support Vector Machines don't provide a direct way to interpret feature importance
        # However, we can visualize the actual vs predicted plot, or use other methods like SHAP for feature importance.
        print("SVM doesn't directly provide feature importance like Random Forest. You may want to use SHAP values for that.")
    
    def run(self):
        # Execute the entire workflow
        self.load_and_clean_data()
        self.prepare_data()
        self.train_model()
        self.evaluate_model()
        self.plot_feature_importance()


# Example usage
if __name__ == "__main__":
    data_path = r'C:\Users\pubgo\OneDrive\Desktop\BSU RAK 24-25\CODE LAB II\SA2-jaydeeci\sports_car_price.csv'
    model_svm = SportsCarPricePredictionSVM(data_path)
    model_svm.run()





