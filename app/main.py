# import libraries and helper files
import os
import numpy as np
import pandas
from flask import Flask, Blueprint, flash, g, redirect, render_template
from flask import request, session, url_for, jsonify
import openai
import inspect
from subprocess import Popen, PIPE
from io import StringIO, BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import base64
import sys
from PIL import Image
try:
    from helper_code import text_code_dict
except ImportError:
    from .helper_code import text_code_dict


'''
HELPER FUNCTIONS
'''
# helper function for parsing code via reference to dictionary
def funcparser(txt):
    code = inspect.getsource(text_code_dict[txt]).replace('    ', '')
    code = code.split('\n', 2)[2]
    return code


# store normal stdout in variable for reference
old_stdout = sys.stdout


# helper function for running code stored in dictionary
def runcode(text):
    # turn off plotting and run function, try to grab fig and save in buffer
    plt.ioff()
    text_code_dict[text]()
    fig = plt.gcf()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close()
    pic = Image.open(buf)
    pix = np.array(pic.getdata(), dtype=np.uint8).reshape(pic.size[1], pic.size[0], -1)
    # if min and max colors are the same, it wasn't a plot - re-run as string
    if np.min(pix) == np.max(pix):
        new_stdout = StringIO()
        sys.stdout = new_stdout
        text_code_dict[text]()
        output = new_stdout.getvalue()
        sys.stdout = old_stdout
        outputtype = 'string'
        sys.stdout = old_stdout
        return [outputtype, output]
    # if it was a plot, then output as HTML image from buffer
    else:
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        output = f"<img src='data:image/png;base64,{data}'/>"
        outputtype = 'image'
        return [outputtype, output]


'''
FLASK APPLICATION CODE & ROUTES
'''
# set up flask application
app = Flask(__name__)


# base route to display main html body
@app.route('/', methods=["GET", "POST"])
def home():
    return render_template('icoder.html')


# run app in debug mode
if __name__ == "__main__":
    app.run(debug=True)


# create a function to read form inputs and process a set of outpts
# returns a json object containing:
    # 'txtcmd': text command
    # 'rawcode': stripped and formatted code text
    # 'output': console output of the code that was run
@app.route('/process')
def process():
    command = request.args.get('command')

    if command not in text_code_dict.keys():
        return 0

    codeblock = funcparser(command.lower())
    [outputtype, output] = runcode(command.lower())
    outputs = [outputtype, command, codeblock, output]

    return jsonify(outputs=outputs)
