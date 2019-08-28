import sys
import os
from flask import Flask, render_template, request, redirect, url_for, Response, flash, request, url_for
from flask.views import View
from flask import Flask, render_template, request
from functools import update_wrapper
from flask_restful import Resource, Api
from flask import Flask, jsonify, render_template, request
import webbrowser
import time
import json
import os
from flask import Flask, render_template, request
from flask import Flask, request, render_template
import numpy as np
from copy import deepcopy
from mnet import *
from code.main import Optimize
import threading
import time
import time
import sys
import requests
import pyprind
from flask import Flask
from flask_mail import Mail, Message
import smtplib
import sys
import os
from flask import Flask
from flask_mail import Mail, Message
import smtplib
import sys
import os
import pylab as plt
import matplotlib.pyplot as plt


sys.path.append('C:/Users/ALEXANDRA/workspace/molnet-master/files')


app = Flask(__name__, instance_path=os.path.join(
    os.path.abspath(os.curdir), 'instance'), instance_relative_config=True)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = 'alexandra.oduron@gmail.com'
app.config['MAIL_PASSWORD'] = 'pavilion26A'
mail = Mail(app)


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=['POST'])
def upload():
    print("I accessed upload")

    target = os.path.join(APP_ROOT, 'files/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)

    return render_template("selection.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/my-link', methods=['POST'])
def my_link():

    start_ms2_vals = int(request.form['text'])
    print(start_ms2_vals)
    stop_ms2_vals = int(request.form['text1'])
    print(stop_ms2_vals)
    step_ms2_vals = int(request.form['text2'])
    print(step_ms2_vals)

    input_list_matched_peaks = str(request.form['text3'])
    P = []
    P = input_list_matched_peaks.split(',')
    new_list1 = []
    for i in P:
        new_list1.append(float(i))
    matched_peaks_array = np.array(new_list1)
    print("I could convert to a np.array FOR matched_peaks")
    print(matched_peaks_array)

    # List input for MZ tolerance
    input_list = str(request.form['text6'])
    L = []
    L = input_list.split(',')
    new_list = []
    for item in L:
        new_list.append(float(item))
    tol_vals_array = np.array(new_list)
    print("I could convert to a np.array")
    print(tol_vals_array)

    d = Optimize()
    d.main(start_ms2_vals, stop_ms2_vals, step_ms2_vals,
           matched_peaks_array, tol_vals_array)

    d.stop_parameters_returning()
    d.get_OptimalParemeters()
    d.get_BestThreshold()
    d.get_mp()
    d.get_m()
    d.get_t()
    min_peaks_values = str(d.get_mp())
    tol_values = str(d.get_t())
    ms2_values = str(d.get_m())

    msg = Message('Your results are: ' 'Minimum number of matched peaks:' + min_peaks_values + 'MS2 tolerance:' + tol_values + 'Minimum MS2 intensity:' + ms2_values, sender='contact@monlet.com',
                  recipients=['2412650o@student.gla.ac.uk'])

    mail.send(msg)

    graph1_url = d.print_graph()
    graph2_url = d.print_graph2()

    print("it worked")
    return render_template("results.html", OptimalParemeters=d.get_OptimalParemeters(), BestThreshold=d.get_BestThreshold(), min_peaks=d.get_mp(), tol_vals=d.get_t(), ms2_vals=d.get_m(), graph1=graph1_url, graph2=graph2_url)


@app.route('/_stuff', methods=['GET'])
def stuff():

    return jsonify(time.time())


if __name__ == "__main__":
    app.run(port=4555, debug=True)
