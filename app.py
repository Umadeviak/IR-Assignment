#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:57:29 2021

@author: umadevi
"""
import time

from flask import Flask, render_template, request
import docSearch
app = Flask(__name__)

@app.route('/')
def results():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process_res():
    start = time.process_time()
    user_search_query = request.form['msg']
    result_list1 = docSearch.get_similarity(user_search_query)
    time_taken = round((time.process_time() - start) * 1000)

    return render_template('index.html', result_list = result_list1, time_taken = time_taken,
                                          user_query=user_search_query
                                          )


if __name__ == "__main__":
    app.run(debug=True)