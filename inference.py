import numpy as np
import pandas as pd

# функция для генерации последовательностей для обучения модели с необходимым шагом (временным лагом)
def create_sequences(values, time_steps):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

def model_inference(df, load_model, StSc, UCL = 7.8, N_STEPS = 60):
    # прогноз на всей выборке и построение невязки
    X = create_sequences(StSc.transform(df), N_STEPS)
    cnn_residuals = pd.Series(np.sum(np.mean(np.abs(X - load_model.predict(X)), axis=1), axis=1))

    # выделение аномалий и разметка на нормальный и аномальный режимы
    anomalous_data = cnn_residuals > UCL
    anomalous_data_indices = []
    for data_idx in range(N_STEPS - 1, len(X) - N_STEPS + 1):
        if np.all(anomalous_data[data_idx - N_STEPS + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)

    prediction = pd.Series(data=0, index=df.index)
    prediction.iloc[anomalous_data_indices] = 1

    # обнаружение точек изменения состояния
    # prediction_cp = abs(prediction.diff())
    # prediction_cp[0] = prediction[0]
    return prediction