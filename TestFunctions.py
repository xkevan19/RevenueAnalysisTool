import os
import unittest
import matplotlib.pyplot as plt
import pandas as pd
from prettytable import PrettyTable
from RevenueAnalysis import load_and_process_data, train_model, predict_new_data, RevenueAnalysis


class TestFunctional(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {'Test Name': [], 'Result': []}

    def add_result(self, test_name, result):
        self.results['Test Name'].append(test_name)
        self.results['Result'].append(result)

    def test_load_valid_csv(self):
        print("\nTesting loading a valid CSV file...")
        # Test loading a valid CSV file
        file_path = 'revenue.csv'
        cleaned_df, _, _, _, _, _ = load_and_process_data(file_path)

        # Assert that the DataFrame is not empty
        if not cleaned_df.empty:
            print("Load valid CSV file test passed.")
            self.add_result("load_valid_csv", "Passed")
        else:
            self.fail("Load valid CSV file test failed: DataFrame is empty.")
            self.add_result("load_valid_csv", "Failed")

        # Add more assertions as needed to verify data processing

    def test_load_invalid_csv(self):
        print("\nTesting loading an invalid CSV file...")
        # Test loading an invalid CSV file
        file_path = 'invalid.csv'

        # Assert that loading raises an IOError
        try:
            load_and_process_data(file_path)
        except IOError:
            print("Load invalid CSV file test passed.")
            self.add_result("load_invalid_csv", "Passed")
            return

        self.fail("Line 26 'file_path = ' must be an invalid file")
        self.add_result("load_invalid_csv", "Failed")

        # Add more assertions as needed to verify error handling

    def test_model_training(self):
        print("\nTesting model training...")
        # Test model training
        # Generate dummy training data
        X_train = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6]})
        y_train = pd.Series([10, 20, 30])

        # Train the model
        model = train_model(X_train, y_train)

        # Assert that the model is not None
        if model is not None:
            print("Model training test passed.")
            self.add_result("model_training", "Passed")
        else:
            self.fail("Model training test failed: Model is None.")
            self.add_result("model_training", "Failed")

        # Add more assertions as needed to verify model training

    def test_prediction(self):
        print("\nTesting prediction on new data...")
        # Test prediction on new data
        # Generate dummy new data
        new_data = pd.DataFrame({'Feature1': [4, 5], 'Feature2': [7, 8]})

        # Generate a dummy model
        class DummyModel:
            @staticmethod
            def predict(_):
                return [40, 50]  # Dummy predictions

        # Predict on new data using the dummy model
        predictions = predict_new_data(DummyModel(), new_data)

        # Assert that the predictions match the expected values
        if predictions == [40, 50]:
            print("Prediction test passed.")
            self.add_result("prediction", "Passed")
        else:
            self.fail("Prediction test failed: Predictions do not match expected values.")
            self.add_result("prediction", "Failed")

        # Add more assertions as needed to verify predictions

    def test_visualization(self):
        print("\nTesting visualization functions...")
        # Test visualization functions
        # Generate dummy data for visualization
        data = pd.DataFrame({'Actual': [10, 20, 30], 'Predicted': [15, 25, 35]})

        # Test scatter plot
        plt.figure()
        plt.scatter(data['Actual'], data['Predicted'])
        plt.title('Scatter Plot')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.close()

        # Test histogram
        plt.figure()
        plt.hist(data['Actual'] - data['Predicted'], bins=10)
        plt.title('Histogram')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.close()

        print("Visualization test passed.")
        self.add_result("visualization", "Passed")

    def test_gui_functionality(self):
        print("\nTesting GUI functionality...")
        try:
            revenue_analysis = RevenueAnalysis()
            revenue_analysis.main()  # Call the main method to simulate GUI interaction
            print("GUI functionality test passed.")
            self.add_result("gui_functionality", "Passed")
        except Exception as e:
            self.fail(f"GUI functionality test failed: {e}")
            self.add_result("gui_functionality", "Failed")


if __name__ == '__main__':
    # Run the tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFunctional)
    result = unittest.TextTestRunner(stream=open(os.devnull, 'w'), verbosity=2).run(suite)
    test_result = "Passed" if result.wasSuccessful() else "Failed"
    print("\nTest Result:", test_result)

    # Calculate pass rate
    total_tests_run = result.testsRun
    passed_tests = total_tests_run - len(result.failures) - len(result.errors)
    pass_rate = (passed_tests / total_tests_run) * 100

    # Print results table
    table = PrettyTable()
    table.field_names = ["Test Name", "Result"]
    for test, result in zip(TestFunctional.results['Test Name'], TestFunctional.results['Result']):
        table.add_row([test, result])

    # Include failed tests in the results table
    if isinstance(result, unittest.TestResult) and not result.wasSuccessful():
        failed_tests = [test.id() for test, _ in result.failures]
        for test in failed_tests:
            table.add_row([test, "Failed"])

    print(table)
    print("\nPass Rate:", "{:.2f}%".format(pass_rate))
