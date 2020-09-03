import time
# from flask import Flask, request
import flask
from flask import request, Response, render_template

import os
import subprocess
import sys
import pandas
from shelljob import proc

import csv
import json
import pandas as pd
from copy import deepcopy


app = flask.Flask(__name__)


@app.route('/stream/',methods=['GET', 'POST'])
def stream():
    def read_process():
        arr = ["18","2","3","9","10"]
        for i in arr:
            time.sleep(1)
            print(i)
            yield i + '\n'
            # yield i
    
    sentence = request.args.get('sentence')
    # sentence = "happy"


    prefix= "python main.py --all --config config.yml --type \""
    cmdEvaluation = prefix + sentence + "\""
    # cmdEvaluation = "python main.py --train --all --config config.yml"

    def inner():
        proc = subprocess.Popen(
            [cmdEvaluation],             #call something with a lot of output so we can see it
            shell=True,
            stdout=subprocess.PIPE,
            universal_newlines=True # so fucking important....
        )

        for line in iter(proc.stdout.readline,''):
            # time.sleep(1)                           # Don't need this just shows the text streaming
            print(line.rstrip())
            yield line.rstrip() + '\n'
    
    resp = flask.Response(inner(),
        mimetype='text/plain'
    )
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Credentials'] = 'false'
    resp.headers['Access-Control-Allow-Headers:'] = 'Content-Type,Connection,Server,Date'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, HEAD, POST, PUT, DELETE, CONNECT, OPTIONS, TRACE, PATCH'
    resp.headers['X-Content-Type-Options'] = 'nosniff'

    resp.headers['Vary'] = '*'
    resp.headers['Accept-encoding'] = 'identity'
    resp.headers['Content-encoding'] = 'identity'
    resp.headers['Content-Encoding'] = 'compress'
    resp.headers['Transfer-encoding'] = 'identity'
    resp.headers['X-Powered-By'] = 'Express'

    return resp




@app.route('/')
def index():
    # render the template (below) that will use JavaScript to read the stream
    return render_template('index.html')

@app.route('/dataS')
def dataS():
    print(os.getcwd())
    path=os.getcwd()+'/data/output/SupervisorAgent_sentence_aggregate.csv'
    print(path)
    csv_col_name = list(pd.read_csv(path).columns)  # 取到列名
    dict_csv_data = csv.DictReader(open(path, 'r'), csv_col_name)  # 以字典方式读取csv数据

    csv_col_name[0] = "sentence"
    csv_col_name[csv_col_name.index("sentence.id")] = "id"

    csv_col_name[csv_col_name.index("gaze.shift")] = "gazeShift"
    csv_col_name[csv_col_name.index("immediate.bs")] = "imBs"
    csv_col_name[csv_col_name.index("delayed.bs")] = "dlBs"
    csv_col_name[csv_col_name.index("gaze.keyboard.ratio")] = "gazeRatio"
    csv_col_name[csv_col_name.index("fix.count")] = "fixNum"
    csv_col_name[csv_col_name.index("correct.error")] = "corErr"
    csv_col_name[csv_col_name.index("uncorrected.error")] = "unErr"
    csv_col_name[csv_col_name.index("fix.duration")] = "fixDur"
    
    csv_col_name[csv_col_name.index("iki.1")] = "ikiSD"
    csv_col_name[csv_col_name.index("wpm.1")] = "wpmSD"
    csv_col_name[csv_col_name.index("bs.1")] = "bsSD"
    csv_col_name[csv_col_name.index("gaze.shift.1")] = "gazeShiftSD"
    csv_col_name[csv_col_name.index("immediate.bs.1")] = "imBsSD"
    csv_col_name[csv_col_name.index("delayed.bs.1")] = "dlBsSD"
    csv_col_name[csv_col_name.index("gaze.keyboard.ratio.1")] = "gazeRatioSD"
    csv_col_name[csv_col_name.index("fix.count.1")] = "fixNumSD"
    csv_col_name[csv_col_name.index("correct.error.1")] = "corErrSD"
    csv_col_name[csv_col_name.index("uncorrected.error.1")] = "unErrSD"
    csv_col_name[csv_col_name.index("fix.duration.1")] = "fixDurSD"

    print(csv_col_name)

    next(dict_csv_data)  # 取出第一条数据也就是表头
    next(dict_csv_data)  # 取出第一条数据也就是表头
    next(dict_csv_data)  # 取出第一条数据也就是表头
    dict1 = {}  # 字典1 整个字典
    dict2 = {}  # 字典2 存的是字典1的对应key的value值

    for rows in dict_csv_data:
        dict2 = deepcopy(rows)  # 复制他
        # dict2.pop('sentence.id')  # 删除key为‘sentence.id’的键值对, keep another one
        dict2.pop('sentence.id.1')
        dict2.pop('agent.id')
        dict2.pop('agent.id.1')
        print(rows['id'])
        dict1[rows['id']] = dict2 # sentence.id is the key of each row: 1,2,3...

    return dict1

    # return "100"


@app.route('/dataT')
def dataT():
    print(os.getcwd())
    path=os.getcwd()+'/data/output/SupervisorAgent_sentence_test.csv'
    print(path)
    csv_col_name = list(pd.read_csv(path).columns)  # 取到列名
    dict_csv_data = csv.DictReader(open(path, 'r'), csv_col_name)  # 以字典方式读取csv数据

    print(csv_col_name)

    csv_col_name[csv_col_name.index("sentence.id")] = "sid"
    csv_col_name[csv_col_name.index("agent.id")] = "aid"
    csv_col_name[csv_col_name.index("target.sentence")] = "sentence"

    csv_col_name[csv_col_name.index("gaze.shift")] = "gazeShift"
    csv_col_name[csv_col_name.index("immediate.bs")] = "imBs"
    csv_col_name[csv_col_name.index("delayed.bs")] = "dlBs"
    csv_col_name[csv_col_name.index("gaze.keyboard.ratio")] = "gazeRatio"
    csv_col_name[csv_col_name.index("fix.count")] = "fixNum"
    csv_col_name[csv_col_name.index("correct.error")] = "corErr"
    csv_col_name[csv_col_name.index("uncorrected.error")] = "unErr"
    csv_col_name[csv_col_name.index("fix.duration")] = "fixDur"
    
    print(csv_col_name)

    next(dict_csv_data)  # 取出第一条数据也就是表头
    dict1 = {}  # 字典1 整个字典
    dict2 = {}  # 字典2 存的是字典1的对应key的value值
    dict3 = {}

    for rows in dict_csv_data:
        dict2 = deepcopy(rows)  # 复制他
        # dict2.pop('sid')  # 删除key为‘sentence.id’的键值对, keep another one
        if dict1.get(rows['sid']): # if sid exist, push the new trial
            dict1[rows['sid']][rows['aid']] = dict2
        else:
            dict1[rows['sid']] = {}
            dict1[rows['sid']][rows['aid']] = dict2

    return dict1

@app.route('/dataG')
def dataG():
    print(os.getcwd())
    path=os.getcwd()+'/data/output/SupervisorAgent_sentence_test.csv'
    print(path)
    csv_col_name = list(pd.read_csv(path).columns)  # 取到列名
    dict_csv_data = csv.DictReader(open(path, 'r'), csv_col_name)  # 以字典方式读取csv数据

    print(csv_col_name)

    csv_col_name[csv_col_name.index("sentence.id")] = "sid"
    csv_col_name[csv_col_name.index("agent.id")] = "aid"
    csv_col_name[csv_col_name.index("target.sentence")] = "sentence"

    csv_col_name[csv_col_name.index("gaze.shift")] = "gazeShift"
    csv_col_name[csv_col_name.index("immediate.bs")] = "imBs"
    csv_col_name[csv_col_name.index("delayed.bs")] = "dlBs"
    csv_col_name[csv_col_name.index("gaze.keyboard.ratio")] = "gazeRatio"
    csv_col_name[csv_col_name.index("fix.count")] = "fixNum"
    csv_col_name[csv_col_name.index("correct.error")] = "corErr"
    csv_col_name[csv_col_name.index("uncorrected.error")] = "unErr"
    csv_col_name[csv_col_name.index("fix.duration")] = "fixDur"
    
    print(csv_col_name)

    next(dict_csv_data)  # 取出第一条数据也就是表头
    dict1 = {}  # 字典1 整个字典
    dict2 = {}  # 字典2 存的是字典1的对应key的value值
    dict3 = {}

    dictResult = {}

    iki = []
    wpm = []
    bs = []
    imBs = []
    dlBs = []
    gazeShift = []
    gazeRatio = []
    fixNum = []
    fixDur = []
    corErr = []
    unErr = []
    

    for rows in dict_csv_data:
        iki.append(float(rows["iki"]))
        wpm.append(float(rows["wpm"]))
        bs.append(float(rows["bs"]))
        imBs.append(float(rows["imBs"]))
        dlBs.append(float(rows["dlBs"]))
        gazeShift.append(float(rows["gazeShift"]))
        gazeRatio.append(float(rows["gazeRatio"]))
        fixNum.append(float(rows["fixNum"]))
        fixDur.append(float(rows["fixDur"]))
        corErr.append(float(rows["corErr"]))
        unErr.append(float(rows["unErr"]))

        dict2 = deepcopy(rows)  # 复制他
        if dict1.get(rows['sid']): # if sid exist, push the new trial
            dict1[rows['sid']][rows['aid']] = dict2
        else:
            dict1[rows['sid']] = {}
            dict1[rows['sid']][rows['aid']] = dict2
    

    import numpy as np

    dictResult["sNum"] = len(dict1)
    dictResult["tNum"] = len(dict1["1"])
    dictResult["ikiMean"] = np.mean(iki) 
    dictResult["ikiSD"] = np.std(iki) 
    dictResult["wpmMean"] = np.mean(wpm) 
    dictResult["wpmSD"] = np.std(wpm) 
    dictResult["bsMean"] = np.mean(bs) 
    dictResult["bsSD"] = np.std(bs) 
    dictResult["imBsMean"] = np.mean(imBs) 
    dictResult["imBsSD"] = np.std(imBs) 
    dictResult["dlBsMean"] = np.mean(dlBs) 
    dictResult["dlBsSD"] = np.std(dlBs) 
    dictResult["gazeShiftMean"] = np.mean(gazeShift) 
    dictResult["gazeShiftSD"] = np.std(gazeShift) 
    dictResult["gazeRatioMean"] = np.mean(gazeRatio) 
    dictResult["gazeRatioSD"] = np.std(gazeRatio) 
    dictResult["fixNumMean"] = np.mean(fixNum) 
    dictResult["fuxNumSD"] = np.std(fixNum) 
    dictResult["fixDurMean"] = np.mean(fixDur) 
    dictResult["fixDurSD"] = np.std(fixDur) 
    dictResult["corErrMean"] = np.mean(corErr) 
    dictResult["corErrSD"] = np.std(corErr) 
    dictResult["unErrMean"] = np.mean(unErr) 
    dictResult["unErrSD"] = np.std(unErr) 


    print(len(dict1))

    # result = []
    # result.append(len(dict1))

    return dictResult

