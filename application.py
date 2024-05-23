import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import tensorflow as tf
from dash import Dash, dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from keras.models import load_model

# app = Dash()
# server = app.server
app = Dash(__name__)
application = app.server

def prepare_data_and_model(stock_csv, model_path):
    df = pd.read_csv(stock_csv)
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df['Date']

    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

    for i in range(0, len(data)):
        new_data["Date"][i] = data['Date'][i]
        new_data["Close"][i] = data["Close"][i]

    new_data.index = new_data.Date
    new_data.drop("Date", axis=1, inplace=True)

    dataset = new_data.values

    train = dataset[0:987, :]
    valid = dataset[987:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []

    for i in range(60, len(train)):
        x_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = load_model(model_path)

    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    train = new_data[:987]
    valid = new_data[987:]
    valid['Predictions'] = closing_price

    return train, valid

google_train, google_valid = prepare_data_and_model("google_stock_price.csv", "saved_model.h5")
meta_train, meta_valid = prepare_data_and_model("META.csv", "saved_model.h5")


app.layout = html.Div([

    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(id="tabs", children=[

        dcc.Tab(label='Google Stock Data', children=[
            html.Div([
                html.H2("Actual closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=google_train.index,
                                y=google_valid["Close"],
                                mode='markers'
                            )

                        ],
                        "layout": go.Layout(
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                ),
                html.H2("Google Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=google_valid.index,
                                y=google_valid["Predictions"],
                                mode='markers'
                            )

                        ],
                        "layout": go.Layout(
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                )
            ])

        ]),
        dcc.Tab(label='Facebook Stock Data', children=[
            html.Div([
                html.H2("Facebook Actual closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="facebook-Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=meta_train.index,
                                y=meta_valid["Close"],
                                mode='markers'
                            )

                        ],
                        "layout": go.Layout(
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                ),
                html.H2("Facebook Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="facebook Predicted Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=meta_valid.index,
                                y=meta_valid["Predictions"],
                                mode='markers'
                            )

                        ],
                        "layout": go.Layout(
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                )
            ])

        ])
        ]),
]),




if __name__ == '__main__':
    application.run(host='0.0.0.0', port='8080')
    # app.run_server(debug=True)