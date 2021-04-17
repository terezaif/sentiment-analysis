import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

# import plotly.io as pio


from transformers import pipeline

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

sentiment_analysis = pipeline("sentiment-analysis")


SENTIMENTS = ["POSITIVE", "NEGATIVE"]
SCORES = [0, 0]


def get_figure(sentiment, scores):
    return go.Figure(
        [go.Bar(x=SENTIMENTS, y=SCORES)],
        layout=go.Layout(template="simple_white"),
    )


fig = get_figure(SENTIMENTS, SCORES)


app.layout = html.Div(
    [
        html.H2(
            id="title",
            children="Sentiment Analysis with Pretrained Transformers Pipeline",
        ),
        html.A(
            "Huggingface Pipeline",
            href="https://huggingface.co/transformers/main_classes/pipelines.html#transformers.TextClassificationPipeline",
            target="_blank",
        ),
        html.Div(children="Add some text you want the sentiment for!"),
        dcc.Textarea(
            id="textarea-state-example",
            value="",
            style={"width": "100%", "height": 200},
        ),
        html.Button("Submit", id="textarea-state-example-button", n_clicks=0),
        html.Div(id="textarea-state-example-output", style={"whiteSpace": "pre-line"}),
        dcc.Graph(id="bar-chart", figure=fig),
    ]
)


@app.callback(
    [
        Output("textarea-state-example-output", "children"),
        Output("bar-chart", "figure"),
    ],
    Input("textarea-state-example-button", "n_clicks"),
    State("textarea-state-example", "value"),
)
def update_output(n_clicks, value):
    fig = get_figure(SENTIMENTS, SCORES)
    if n_clicks > 0:
        if 0 < len(value) < 2000:
            text, scores = get_sentiment(value)
            fig = get_figure(SENTIMENTS, scores)
            return text, fig
        else:
            return "Please add a text between 0 and 500 characters!", fig
    else:
        return "", fig


def get_sentiment(text):
    sentiments = SENTIMENTS
    scores = SCORES
    result = sentiment_analysis(text)[0]
    sent = result["label"]
    idx = sentiments.index(sent)
    scores[idx] = result["score"]
    scores[(idx + 1) % 2] = 1 - result["score"]
    text = (
        f"The text sentiment is {result['label']} with score {round(result['score'],2)}"
    )
    return text, scores


# Add the server clause:
if __name__ == "__main__":
    app.run_server()
