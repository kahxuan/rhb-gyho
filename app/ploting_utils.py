import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.subplots as sp


data_pattern = {}
for acc_type in ['saving', 'current', 'credit']:
	data_pattern[acc_type] = {
		'amount': pd.read_csv('static/data/pattern/{}/summ_amount.csv'.format(acc_type)),
		'diff': pd.read_csv('static/data/pattern/{}/summ_diff.csv'.format(acc_type))
	}

with open('static/data/algo/votes_combined.json', 'r') as f:
	votes_combined = json.load(f)

df_preds = pd.read_csv('static/data/algo/votes.csv')

df_trans = {}
df_trans_rolled = {}
for acc_type in ['saving', 'current', 'credit']:
	df_trans[acc_type] = pd.read_csv('static/data/algo/transaction/{}/transaction.csv'.format(acc_type))
	df_trans[acc_type]['day'] = pd.to_datetime(df_trans[acc_type]['day'])
	df_trans_rolled[acc_type] = pd.read_csv('static/data/algo/transaction/{}/transaction_rolled.csv'.format(acc_type))
	df_trans_rolled[acc_type]['date'] = pd.to_datetime(df_trans_rolled[acc_type]['date'])


def get_cids():
	return list(votes_combined.keys())


def plot_pattern(cd):
	res = {}
	for acc_type in data_pattern:

		res[acc_type] = {}

		df = data_pattern[acc_type]['amount']
		df = df[df['Type'] == cd]
		fig = px.line(
			df, x="Transaction Amount", y="Frequency", 
			color='Segment after 6 months', line_shape='spline', log_x=True,
			height=190, width=500,
			color_discrete_sequence=['#0664ac', '#e73d40']
		)
		fig.update_layout(
			margin=dict(l=0, r=0, t=0, b=0),
			paper_bgcolor='rgba(0,0,0,0)',
			plot_bgcolor='rgba(0,0,0,0)',
			showlegend=False,
			xaxis=dict(
				 tickmode = 'array',
				 tickvals =  [10**i for i in range(-2, 5)]
			)
		)
		res[acc_type]['amount'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

		df = data_pattern[acc_type]['diff']
		df = df[df['Type'] == cd]
		fig = px.line(
			df, x="First Difference", y="Frequency", 
			color='Segment after 6 months', line_shape='spline', log_x=True,
			height=190, width=500,
			color_discrete_sequence=['#0664ac', '#e73d40'],
			labels={
				'First Difference': 'Inconsistency'
			}
		)
		fig.update_layout(
			margin=dict(l=0, r=0, t=0, b=0),
			paper_bgcolor='rgba(0,0,0,0)',
			plot_bgcolor='rgba(0,0,0,0)',
			showlegend=False,
			xaxis=dict(
				 tickmode = 'array',
				 tickvals =  [10**i for i in range(0, 6)]
			)
		)
		res[acc_type]['diff'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	return res


def plot_voting(cid):
	df = df_preds[df_preds['id'] == cid]
	fig = px.bar(
	    df, x="Account", y="Value", 
	    color="Feature", barmode="group",
	    color_discrete_sequence=['#0664ac', '#6dc7dc'],
	    height=350
	)
	fig.update_layout(
		margin=dict(l=0, r=0, t=0, b=0),
		paper_bgcolor='rgba(0,0,0,0)',
		plot_bgcolor='rgba(0,0,0,0)',
		legend=dict(
		    orientation="h",
		    yanchor="bottom",
		    y=1.02,
		    xanchor="left",
		    x=0
		)
	)
	return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def plot_feature_extraction(cid, acc_type):

	colors = ['#0664ac', '#e73d40']


	df1 = df_trans[acc_type][df_trans[acc_type]['id'] == cid]

	if df1['value'].sum() == 0:
		return None

	df2 = df_trans_rolled[acc_type][df_trans_rolled[acc_type]['id'] == cid]

	fig1 = px.line(df1.sort_values('day'), x="day", y="value", color='type', color_discrete_sequence=colors)
	fig2 = px.line(df2.sort_values('date'), x="date", y="value", color='type', color_discrete_sequence=colors)

	df2.loc[:, 'value'] = np.log(df2['value'][df2['value'] > 0])
	fig3 = px.histogram(df2, x="value", color='type',  nbins=30, opacity=0.75, color_discrete_sequence=colors)

	titles = [
	    'Transaction Amount Per Day',
	    'Transaction Amount Per Day (Smoothed)',
	    'Frequencies of Transaction Amount'
	]

	fig1_traces = []
	fig2_traces = []
	fig3_traces = []
	for trace in range(len(fig1["data"])):
		fig1_traces.append(fig1["data"][trace])
	for trace in range(len(fig2["data"])):
		fig2_traces.append(fig2["data"][trace])
	for trace in range(len(fig3["data"])):
		fig3_traces.append(fig3["data"][trace])


	fig = sp.make_subplots(rows=3, cols=1, subplot_titles=titles) 


	for traces in fig1_traces:
	    fig.append_trace(traces, row=1, col=1)
	for traces in fig2_traces:
	    fig.append_trace(traces, row=2, col=1)
	for traces in fig3_traces:
	    fig.append_trace(traces, row=3, col=1)

	    
	for i in range(len(fig.data) - 2):
	    fig.data[i]['showlegend'] = False

	fig.update_layout(
	    margin=dict(l=0, r=0, t=0, b=0),
	    paper_bgcolor='rgba(0,0,0,0)',
	    plot_bgcolor='rgba(0,0,0,0)',
	    legend=dict(
	        orientation="h",
	        yanchor="bottom",
	        y=1.02,
	        xanchor="left",
	        x=0,
	    ),
	    height=700,
	)
	fig['layout']['xaxis1']['title']='Date'
	fig['layout']['xaxis2']['title']='Date'
	fig['layout']['xaxis3']['title']='Transaction Amount (log transformed)'
	fig['layout']['yaxis1']['title']='Amount'
	fig['layout']['yaxis2']['title']='Amount'
	fig['layout']['yaxis3']['title']='Frequency'
	return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def get_predictions(cid):
    return round(votes_combined[str(cid)] * 100, 2)



