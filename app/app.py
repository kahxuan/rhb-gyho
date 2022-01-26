from flask import Flask, render_template
import pandas as pd
import json
import plotly
import plotly.express as px

app = Flask(__name__)

@app.route('/')
def main():
   return render_template(
      'main.html', 
      graphJSON=plot_placeholder(),
      graphJSON2=plot_placeholder(),
      graphJSON3=plot_placeholder3()
   )


def plot_placeholder():
   df = pd.DataFrame({
      'Fruit': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 
      'Bananas'],
      'Amount': [4, 1, 2, 2, 4, 5],
      'City': ['SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal']
   })
   fig = px.bar(df, x='Fruit', y='Amount', color='City', 
      barmode='group', height=220)
   fig.update_layout(
      margin=dict(l=0, r=0, t=0, b=0),
      paper_bgcolor='rgba(0,0,0,0)',
      # plot_bgcolor='rgba(0,0,0,0)'
   )
   graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
   return graphJSON

def plot_placeholder2():
   df = pd.DataFrame({
      'Fruit': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 
      'Bananas'],
      'Amount': [4, 1, 2, 2, 4, 5],
      'City': ['SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal']
   })
   fig = px.bar(df, x='Fruit', y='Amount', color='City', 
      barmode='group', height=300)
   fig.update_layout(
      margin=dict(l=0, r=0, t=0, b=0),
      paper_bgcolor='rgba(0,0,0,0)',
      # plot_bgcolor='rgba(0,0,0,0)'
   )
   graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
   return graphJSON

def plot_placeholder3():
   df = pd.DataFrame({
      'Fruit': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 
      'Bananas'],
      'Amount': [4, 1, 2, 2, 4, 5],
      'City': ['SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal']
   })
   fig = px.bar(df, x='Fruit', y='Amount', color='City', 
      barmode='group', height=200, width=500)
   fig.update_layout(
      margin=dict(l=0, r=0, t=0, b=0),
      paper_bgcolor='rgba(0,0,0,0)',
      # plot_bgcolor='rgba(0,0,0,0)'
   )
   graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
   return graphJSON


app.run(debug=True)