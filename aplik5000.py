import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import os

class LottoPredictionApp:
    MIN_SAMPLES_REQUIRED = 50  # Minimalna liczba próbek wymagana do trenowania modeli

    def __init__(self, root):
        self.root = root
        self.root.title("Lotto Prediction App")
        self.xgb_model = None
        self.lstm_model = None
        self.classification_model = None
        self.data = None

        # GUI elements
        self.load_data_btn = tk.Button(root, text="Załaduj Dane", command=self.load_data)
        self.load_data_btn.pack(pady=10)

        self.select_data_range_btn = tk.Button(root, text="Wybierz Zakres Danych", command=self.select_data_range, state=tk.DISABLED)
        self.select_data_range_btn.pack(pady=10)

        self.train_xgb_model_btn = tk.Button(root, text="Trenuj Model XGBoost", command=self.train_xgb_model, state=tk.DISABLED)
        self.train_xgb_model_btn.pack(pady=10)

        self.train_lstm_model_btn = tk.Button(root, text="Trenuj Model LSTM", command=self.train_lstm_model, state=tk.DISABLED)
        self.train_lstm_model_btn.pack(pady=10)

        self.train_classification_model_btn = tk.Button(root, text="Trenuj Model Klasyfikacyjny", command=self.train_classification_model, state=tk.DISABLED)
        self.train_classification_model_btn.pack(pady=10)

        self.search_data_btn = tk.Button(root, text="Znajdź Optymalny Zakres Danych", command=self.search_data_range, state=tk.DISABLED)
        self.search_data_btn.pack(pady=10)

        self.generate_numbers_btn = tk.Button(root, text="Wygeneruj Liczby", command=self.generate_numbers, state=tk.DISABLED)
        self.generate_numbers_btn.pack(pady=10)

        self.update_data_btn = tk.Button(root, text="Aktualizuj Dane", command=self.update_data, state=tk.NORMAL)
        self.update_data_btn.pack(pady=10)

        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(pady=10)

        self.output_label = tk.Label(root, text="Predykcja Liczb: ")
        self.output_label.pack(pady=20)

        self.data_range = None

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.file_path = file_path
                self.data = pd.read_csv(self.file_path, sep=';')
                if self.validate_data():
                    self.data = self.feature_engineering(self.data)
                    messagebox.showinfo("Informacja", "Pomyślnie załadowano i przygotowano dane.")
                    self.select_data_range_btn.config(state=tk.NORMAL)
                    self.search_data_btn.config(state=tk.NORMAL)
                else:
                    self.data = None  # Reset danych w przypadku błędu
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się załadować danych: {e}")

    def validate_data(self):
        if self.data is not None:
            # Sprawdź, czy wszystkie wymagane kolumny są obecne
            required_columns = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
            if not all(column in self.data.columns for column in required_columns):
                messagebox.showerror("Błąd", "Dane nie zawierają wszystkich wymaganych kolumn.")
                return False

            # Sprawdź, czy nie ma brakujących wartości
            if self.data[required_columns].isnull().values.any():
                messagebox.showerror("Błąd", "Dane zawierają brakujące wartości.")
                return False

            # Sprawdź, czy liczby są w odpowiednim zakresie
            if not self.data[required_columns].applymap(lambda x: 1 <= x <= 49).all().all():
                messagebox.showerror("Błąd", "Dane zawierają liczby poza zakresem 1-49.")
                return False

            return True
        else:
            messagebox.showerror("Błąd", "Dane nie zostały załadowane.")
            return False

    def feature_engineering(self, data):
        # Dodaj kolumnę z sumą liczb
        data['Sum'] = data[['L1', 'L2', 'L3', 'L4', 'L5', 'L6']].sum(axis=1)
        # Dodaj kolumnę z średnią liczb
        data['Mean'] = data[['L1', 'L2', 'L3', 'L4', 'L5', 'L6']].mean(axis=1)
        # Dodaj kolumnę z odchyleniem standardowym
        data['Std'] = data[['L1', 'L2', 'L3', 'L4', 'L5', 'L6']].std(axis=1)
        return data

    def select_data_range(self):
        if self.data is None:
            messagebox.showerror("Błąd", "Najpierw załaduj dane.")
            return

        total_samples = len(self.data)
        min_samples_required = self.MIN_SAMPLES_REQUIRED

        start = simpledialog.askinteger("Zakres Danych", f"Podaj początkowy indeks danych do szkolenia (0 - {total_samples - 1}):")
        if start is None:
            return
        end = simpledialog.askinteger("Zakres Danych", f"Podaj końcowy indeks danych do szkolenia ({start} - {total_samples - 1}):")
        if end is None:
            return

        if start < 0 or end >= total_samples or start > end:
            messagebox.showerror("Błąd", "Nieprawidłowy zakres danych. Upewnij się, że start ≤ end i indeksy są w odpowiednim zakresie.")
            return

        num_samples = end - start + 1
        if num_samples < min_samples_required:
            messagebox.showerror("Błąd", f"Wybrany zakres danych zawiera {num_samples} próbek. Wymagane jest co najmniej {min_samples_required} próbek.")
            return

        self.data_range = (start, end + 1)  # Dodajemy 1 do end, ponieważ w iloc ostatni indeks jest wyłączony
        messagebox.showinfo("Informacja", f"Wybrano zakres danych: od {start} do {end}.")
        self.train_xgb_model_btn.config(state=tk.NORMAL)
        self.train_lstm_model_btn.config(state=tk.NORMAL)
        self.train_classification_model_btn.config(state=tk.NORMAL)

    def update_data(self):
        if hasattr(self, 'file_path') and os.path.exists(self.file_path):
            try:
                self.data = pd.read_csv(self.file_path, sep=';')
                if self.validate_data():
                    self.data = self.feature_engineering(self.data)
                    messagebox.showinfo("Informacja", "Dane zostały zaktualizowane.")
                else:
                    self.data = None
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się zaktualizować danych: {e}")
        else:
            messagebox.showerror("Błąd", "Plik danych nie jest dostępny.")

    def normalize_data(self, X, y):
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        return X_scaled, y_scaled, scaler_X, scaler_y

    def train_xgb_model(self):
        if self.data is None or self.data_range is None:
            messagebox.showerror("Błąd", "Najpierw załaduj dane i wybierz zakres.")
            return

        self.progress.start()
        try:
            # Preprocessing data
            start, end = self.data_range
            X, y = self.prepare_data(self.data.iloc[start:end])
            X, y, self.scaler_X, self.scaler_y = self.normalize_data(X, y)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            n_samples = X_train.shape[0]
            if n_samples < 5:
                messagebox.showerror("Błąd", f"Zbyt mało danych w zbiorze treningowym ({n_samples} próbek). Wymagane co najmniej 5 próbek.")
                self.progress.stop()
                return

            # Define parameter grid for RandomizedSearchCV
            param_dist = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
            }
            xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            randomized_search = RandomizedSearchCV(
                xgb_reg,
                param_distributions=param_dist,
                n_iter=10,
                cv=min(5, n_samples),
                scoring='neg_mean_squared_error',
                random_state=42
            )
            randomized_search.fit(X_train, y_train)
            self.xgb_model = randomized_search.best_estimator_

            # Kroswalidacja
            kf = KFold(n_splits=min(5, n_samples))
            scores = cross_val_score(self.xgb_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
            mse_scores = -scores
            average_mse = mse_scores.mean()

            # Evaluate the model
            predictions = self.xgb_model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            messagebox.showinfo("Informacja", f"Model XGBoost został wytrenowany.\nMSE: {mse:.4f}\nŚrednie MSE w kroswalidacji: {average_mse:.4f}")
            self.generate_numbers_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił błąd podczas trenowania modelu XGBoost: {e}")
        finally:
            self.progress.stop()

    def train_lstm_model(self):
        if self.data is None or self.data_range is None:
            messagebox.showerror("Błąd", "Najpierw załaduj dane i wybierz zakres.")
            return

        self.progress.start()
        try:
            # Preprocessing data
            start, end = self.data_range
            X, y = self.prepare_data(self.data.iloc[start:end], lstm=True)
            X_shape = X.shape
            X = X.reshape(-1, X.shape[-1])
            X, y, self.scaler_X_lstm, self.scaler_y_lstm = self.normalize_data(X, y)
            X = X.reshape(X_shape)

            n_samples = X.shape[0]
            if n_samples < 5:
                messagebox.showerror("Błąd", f"Zbyt mało danych do trenowania modelu LSTM ({n_samples} próbek). Wymagane co najmniej 5 próbek.")
                self.progress.stop()
                return

            # Build the LSTM model
            self.lstm_model = Sequential()
            self.lstm_model.add(Input(shape=(X.shape[1], X.shape[2])))
            self.lstm_model.add(LSTM(128, activation='relu', return_sequences=True))
            self.lstm_model.add(LSTM(64, activation='relu'))
            self.lstm_model.add(Dense(6))
            self.lstm_model.compile(optimizer='adam', loss='mse')

            # Train the LSTM model
            self.lstm_model.fit(X, y, epochs=50, batch_size=16, verbose=0)

            messagebox.showinfo("Informacja", "Model LSTM został wytrenowany.")
            self.generate_numbers_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił błąd podczas trenowania modelu LSTM: {e}")
        finally:
            self.progress.stop()

    def train_classification_model(self):
        if self.data is None or self.data_range is None:
            messagebox.showerror("Błąd", "Najpierw załaduj dane i wybierz zakres.")
            return

        self.progress.start()
        try:
            start, end = self.data_range
            X, y = self.prepare_data_classification(self.data.iloc[start:end])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            n_samples = X_train.shape[0]
            if n_samples < 5:
                messagebox.showerror("Błąd", f"Zbyt mało danych w zbiorze treningowym ({n_samples} próbek). Wymagane co najmniej 5 próbek.")
                self.progress.stop()
                return

            base_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            clf = MultiOutputClassifier(base_clf)
            clf.fit(X_train, y_train)
            self.classification_model = clf

            # Ocena modelu
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)

            train_hamming = hamming_loss(y_train, y_pred_train)
            test_hamming = hamming_loss(y_test, y_pred_test)
            f1 = f1_score(y_test, y_pred_test, average='micro')

            messagebox.showinfo("Informacja", f"Model klasyfikacyjny został wytrenowany.\nHamming Loss na zbiorze testowym: {test_hamming:.4f}\nF1-score: {f1:.4f}")
            self.generate_numbers_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Błąd", f"Wystąpił błąd podczas trenowania modelu klasyfikacyjnego: {e}")
        finally:
            self.progress.stop()

    def prepare_data(self, data, lstm=False):
        numbers = data[['L1', 'L2', 'L3', 'L4', 'L5', 'L6']].values
        additional_features = data[['Sum', 'Mean', 'Std']].values
        X, y = [], []

        if lstm:
            window_size = 5
            for i in range(len(numbers) - window_size):
                X_sample = np.hstack((numbers[i:i + window_size].reshape(-1), additional_features[i + window_size - 1]))
                X.append(X_sample)
                y.append(numbers[i + window_size])
            X = np.array(X)
            y = np.array(y)
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        else:
            for i in range(len(numbers) - 1):
                X_sample = np.hstack((numbers[i], additional_features[i]))
                X.append(X_sample)
                y.append(numbers[i + 1])
            X = np.array(X)
            y = np.array(y)

        return X, y

    def prepare_data_classification(self, data):
        numbers = data[['L1', 'L2', 'L3', 'L4', 'L5', 'L6']].values
        X, y = [], []
        for i in range(len(numbers) - 5):
            X_sample = numbers[i:i+5].flatten()
            y_sample = np.zeros(49)
            for num in numbers[i+5]:
                y_sample[int(num)-1] = 1
            X.append(X_sample)
            y.append(y_sample)
        X = np.array(X)
        y = np.array(y)
        return X, y

    def search_data_range(self):
        if self.data is None:
            messagebox.showerror("Błąd", "Najpierw załaduj dane.")
            return

        target_numbers = simpledialog.askstring("Wyszukiwanie Zakresu Danych", "Podaj liczby docelowe (oddzielone przecinkami, np. 1,2,3,4,5,6):")
        if target_numbers:
            try:
                target_numbers = [int(num.strip()) for num in target_numbers.split(',')]
                if len(target_numbers) != 6:
                    raise ValueError("Należy podać dokładnie 6 liczb.")

                best_range = None
                max_matches = 0

                # Iterate over all possible ranges in the data
                for start in range(len(self.data) - self.MIN_SAMPLES_REQUIRED + 1):
                    end = start + self.MIN_SAMPLES_REQUIRED
                    if end > len(self.data):
                        break  # Nie mamy wystarczającej liczby danych od tego punktu

                    current_numbers = self.data[['L1', 'L2', 'L3', 'L4', 'L5', 'L6']].iloc[start:end].values.flatten()
                    matches = len(set(target_numbers) & set(current_numbers))
                    if matches > max_matches and matches >= 3:
                        max_matches = matches
                        best_range = (start, end)

                if best_range:
                    self.data_range = best_range
                    messagebox.showinfo("Informacja", f"Znaleziono optymalny zakres danych: od {best_range[0]} do {best_range[1]-1}. Liczba trafień: {max_matches}")
                    self.train_xgb_model_btn.config(state=tk.NORMAL)
                    self.train_lstm_model_btn.config(state=tk.NORMAL)
                    self.train_classification_model_btn.config(state=tk.NORMAL)
                else:
                    messagebox.showinfo("Informacja", f"Nie znaleziono zakresu danych spełniającego kryteria z minimalną liczbą {self.MIN_SAMPLES_REQUIRED} próbek i co najmniej 3 trafieniami.")
            except ValueError as e:
                messagebox.showerror("Błąd", str(e))
            except Exception as e:
                messagebox.showerror("Błąd", f"Wystąpił błąd: {e}")

    def generate_numbers(self):
        if self.xgb_model is None and self.lstm_model is None and self.classification_model is None:
            messagebox.showerror("Błąd", "Najpierw wytrenuj przynajmniej jeden model.")
            return

        # Generate prediction based on recent data
        recent_data = self.data[['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'Sum', 'Mean', 'Std']].iloc[-5:].values

        xgb_output = "XGBoost: Model nie jest wytrenowany."
        lstm_output = "LSTM: Model nie jest wytrenowany."
        classification_output = "Klasyfikacja: Model nie jest wytrenowany."

        if self.xgb_model is not None:
            try:
                recent_numbers = self.data[['L1', 'L2', 'L3', 'L4', 'L5', 'L6']].iloc[-1].values
                recent_features = self.data[['Sum', 'Mean', 'Std']].iloc[-1].values
                recent_data_xgb = np.hstack((recent_numbers, recent_features)).reshape(1, -1)
                recent_data_xgb_scaled = self.scaler_X.transform(recent_data_xgb)
                xgb_prediction = self.xgb_model.predict(recent_data_xgb_scaled)
                xgb_prediction_rescaled = self.scaler_y.inverse_transform(xgb_prediction)
                xgb_predicted_numbers = list(set([max(1, min(49, int(round(num)))) for num in xgb_prediction_rescaled.flatten()]))
                while len(xgb_predicted_numbers) < 6:
                    new_num = np.random.randint(1, 50)
                    if new_num not in xgb_predicted_numbers:
                        xgb_predicted_numbers.append(new_num)
                xgb_output = f"XGBoost: {sorted(xgb_predicted_numbers)}"
            except Exception as e:
                xgb_output = f"XGBoost: Błąd podczas generowania liczb ({e})"

        if self.lstm_model is not None:
            try:
                recent_data_lstm = recent_data[:, :-3]  # Pomijamy dodatkowe cechy
                recent_features = recent_data[-1, -3:]  # Ostatnie dodatkowe cechy
                X_input = np.hstack((recent_data_lstm.flatten(), recent_features)).reshape(1, 1, -1)
                X_input_scaled = self.scaler_X_lstm.transform(X_input.reshape(-1, X_input.shape[-1])).reshape(X_input.shape)
                lstm_prediction = self.lstm_model.predict(X_input_scaled)
                lstm_prediction_rescaled = self.scaler_y_lstm.inverse_transform(lstm_prediction)
                lstm_predicted_numbers = list(set([max(1, min(49, int(round(num)))) for num in lstm_prediction_rescaled[0]]))
                while len(lstm_predicted_numbers) < 6:
                    new_num = np.random.randint(1, 50)
                    if new_num not in lstm_predicted_numbers:
                        lstm_predicted_numbers.append(new_num)
                lstm_output = f"LSTM: {sorted(lstm_predicted_numbers)}"
            except Exception as e:
                lstm_output = f"LSTM: Błąd podczas generowania liczb ({e})"

        if self.classification_model is not None:
            try:
                recent_numbers = self.data[['L1', 'L2', 'L3', 'L4', 'L5', 'L6']].iloc[-5:].values.flatten()
                X_input = recent_numbers.reshape(1, -1)
                y_pred_proba = self.classification_model.predict_proba(X_input)
                proba_per_class = np.array([proba[0][1] for proba in y_pred_proba])
                predicted_numbers = np.argsort(proba_per_class)[-6:] + 1
                classification_output = f"Klasyfikacja: {sorted(predicted_numbers)}"
            except Exception as e:
                classification_output = f"Klasyfikacja: Błąd podczas generowania liczb ({e})"

        self.output_label.config(text=f"Predykcja Liczb (pamiętaj o losowości wyników):\n{xgb_output}\n{lstm_output}\n{classification_output}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LottoPredictionApp(root)
    root.mainloop()
