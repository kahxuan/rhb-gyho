from flask import Flask, render_template, request
import pandas as pd
import json
from ploting_utils import (
   plot_pattern, 
   get_cids,
   plot_voting,
   get_predictions,
   plot_feature_extraction
)

app = Flask(__name__, static_url_path='/static')

CID_SELECTED = 15266186
ACC_SELECTED = 'saving'

@app.route('/callback_pattern', methods=['POST', 'GET'])
def cb_pattern():
   return plot_pattern(request.args.get('cd'))


@app.route('/callback_algo', methods=['POST', 'GET'])
def cb_algo():
   cid, acc_type = int(request.args.get('cid')), request.args.get('accType')
   res = {
      'feature': plot_feature_extraction(cid, acc_type),
      'voting': plot_voting(cid),
      'prob': get_predictions(cid)
   }
   
   return res


@app.route('/')
def main():
   return render_template(
      'main.html', 
      graphsPattern=plot_pattern('debit'),
      cids=get_cids(),
      cid_selected=CID_SELECTED,
      graphVotes=plot_voting(CID_SELECTED),
      prediction=get_predictions(CID_SELECTED),
      graphsFeature=plot_feature_extraction(CID_SELECTED, ACC_SELECTED)
   )


app.run(debug=True)