import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class Preprocessor:
    """Custom transformer for data preprocessing tasks"""

    def __init__(self):
        pass

    def transform(self, data):
        """Preprocess the data"""
        X = data.drop(["RainTomorrow"], axis=1)
        # X = data.drop(["RainTomorrow"])
        y = data["RainTomorrow"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test


class ModelBuilder:
    """Build and train the neural network model"""

    def __init__(self):
        pass

    def build_model(self, input_dim):
        """Build the neural network model"""
        model = Sequential()
        model.add(Dense(units=32, kernel_initializer='uniform',
                  activation='relu', input_dim=input_dim))
        model.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(0.5))
        model.add(
            Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
        return model

    def train_model(self, model, X_train, y_train):
        """Train the neural network model"""
        early_stopping = callbacks.EarlyStopping(
            min_delta=0.001,
            patience=20,
            restore_best_weights=True,
        )
        opt = Adam(learning_rate=0.00009)
        model.compile(optimizer=opt, loss='binary_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size=32, epochs=150, callbacks=[
                            early_stopping], validation_split=0.2)
        return history


class Evaluator:
    """Evaluate the trained model"""

    def __init__(self):
        pass

    def plot_loss(self, history):
        """Plot training and validation loss over epochs"""
        history_df = pd.DataFrame(history.history)
        plt.plot(history_df['loss'], "#BDE2E2", label='Training loss')
        plt.plot(history_df['val_loss'], "#C2C4E2", label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc="best")
        plt.show()

    def plot_accuracy(self, history):
        """Plot training and validation accuracy over epochs"""
        history_df = pd.DataFrame(history.history)
        plt.plot(history_df['accuracy'], "#BDE2E2", label='Training accuracy')
        plt.plot(history_df['val_accuracy'], "#C2C4E2",
                 label='Validation accuracy')
        plt.title('Training and Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model on the test set"""
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5)
        cmap1 = sns.diverging_palette(260, -10, s=50, l=75, n=5, as_cmap=True)
        plt.subplots(figsize=(12, 8))
        cf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(cf_matrix / np.sum(cf_matrix), cmap=cmap1,
                    annot=True, annot_kws={'size': 15})
        print(classification_report(y_test, y_pred))
