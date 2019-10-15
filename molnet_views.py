#This class was written by me

# Dependencies
import sys
import os
from flask import Flask, render_template, request, redirect, url_for, Response, flash, request
import numpy as np
from mnet import *
from code.optimize_parameters import Optimize
from flask_mail import Mail, Message
import smtplib
import redis
from celery import Celery


sys.path.append('C:/Users/ALEXANDRA/workspace/molnet-master/files')
# Name app
app = Flask(__name__, instance_path=os.path.join(
    os.path.abspath(os.curdir), 'instance'), instance_relative_config=True)
# Configuration for sending an email with results
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = 'molnet.dissertation@gmail.com'
app.config['MAIL_PASSWORD'] = 'dissertation'
mail = Mail(app)


# Home page
@app.route("/")
def index():
    return render_template("upload.html")

# Upload .mzML file
@app.route("/upload", methods=['POST'])
def upload():
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

# About page
@app.route("/about")
def about():
    return render_template("about.html")

# Get values of parameters and start optimization
@app.route('/optimize', methods=['POST'])
def optimize():

    # Inputs for MS2 intensity: Start, Stop and Step values.
    start_ms2_vals = int(request.form['text'])
    stop_ms2_vals = int(request.form['text1'])
    step_ms2_vals = int(request.form['text2'])
    # List input for Minimum number of matched peaks
    input_list_matched_peaks = str(request.form['text3'])
    # Put data in a np.array
    P = []
    P = input_list_matched_peaks.split(',')
    new_list1 = []
    for i in P:
        new_list1.append(float(i))
    matched_peaks_array = np.array(new_list1)

    # List input for MZ tolerance
    input_list = str(request.form['text6'])
    L = []
    L = input_list.split(',')
    new_list = []
    for item in L:
        new_list.append(float(item))
    # Put data in a np.array
    tol_vals_array = np.array(new_list)

    # Start optimization process
    d = Optimize()
    d.main(start_ms2_vals, stop_ms2_vals, step_ms2_vals,
           matched_peaks_array, tol_vals_array)
    # Thread stop, notify that results are ready
    d.stop_parameters_returning()
    # Get results of the value of the optimized parameters
    d.get_BestThreshold()
    d.get_mp()
    d.get_m()
    d.get_t()
    # Make results a string to display results in the email
    min_peaks_values = str(d.get_mp())
    tol_values = str(d.get_t())
    ms2_values = str(d.get_m())
    # Sent an email with the results

    msg = Message('Parameter Optimization results:',
                  sender='contact@monlet.com', recipients=['2412650o@student.gla.ac.uk'])
    msg.body = "These are your results \n" 'Minimum number of matched peaks:  ' + \
        min_peaks_values + '  MS2 tolerance:  ' + tol_values + \
        '  Minimum MS2 intensity:  ' + ms2_values
    mail.send(msg)

    # Get heat maps
    graph1_url = d.print_graph()
    graph2_url = d.print_graph2()
    # Display results page
    return render_template("results.html", BestThreshold=d.get_BestThreshold(), min_peaks=d.get_mp(), tol_vals=d.get_t(), ms2_vals=d.get_m(), graph1=graph1_url, graph2=graph2_url)


# Create a celery object
def make_celery(app):
    celery = Celery(
        app.name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery


# Celery configuration
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
# Pass the web application through the celery object.
celery = make_celery(app)


@app.route('/test')
def test():
    result = background_task.delay()
    # After doing the test, the variable "result" should be put inside the URL /optimize'
    # and the method optimize(),
    # rigth after the insertion of the values of the parameters in the np.array.
    # The code that initializes the parameter optimization process and the code
    # that sends the email, should be removed.
    return 'This is a text to test that the task was run'


@celery.task(name='molnet_views.background_task')
def background_task():
    print("The parameter optimization process should be run here")
    #d = Optimize()
    # d.main(start_ms2_vals, stop_ms2_vals, step_ms2_vals,
    #       matched_peaks_array, tol_vals_array)
    print("The code that sends the email should be put here")
    # Sent an email with the results
    # msg = Message('Parameter Optimization results:', sender='contact@monlet.com',
    #              recipients=['2412650o@student.gla.ac.uk'])
    # msg.body = "These are your results \n" 'Minimum number of matched peaks:  ' + \
    #    min_peaks_values + '  MS2 tolerance:  ' + tol_values + \
    #    '  Minimum MS2 intensity:  ' + ms2_values
    # mail.send(msg)
    return "Job"


if __name__ == "__main__":
    app.run(port=4555, debug=True)
