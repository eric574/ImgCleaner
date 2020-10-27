import os
from app import app
from os.path import join, dirname, realpath
from flask import request, redirect, url_for, render_template, flash, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import sys
sys.path.append("..")
import denoise_run

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/uploads/')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello(filename="", cleaned_path="", error=""):
    # if request.method == 'POST':
    # if request
        # redirect(request.path)
    return render_template('index.html', filename=filename, cleaned_path=cleaned_path, error=error)

# Route to upload files
@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            f = request.files['file']
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
            # return redirect(url_for('hello', filename=f.filename))
            return render_template('index.html', filename=f.filename)
        except Exception as e:
            return render_template('index.html', error=e)
    return redirect(url_for('hello'))

# @app.route('/show/<filename>')
# def uploaded_file(filename):
#     return render_template('index.html', filename=filename)

@app.route('/uploads/<filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/return_file/<filename>')
def return_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment = True)

@app.route('/denoise/<filename>')
def denoise(filename):
    img = 0
    fp = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(fp)
    if "text" in filename:
        img = denoise_run.detext(fp)
    else:
        img = denoise_run.denoise(fp)
    print(type(img))
    # print(img)
    cleaned_path = UPLOAD_FOLDER + 'cleaned-' + filename
    img.save(cleaned_path)
    print(cleaned_path)
    print(fp)
    return render_template('index.html', filename=filename, cleaned_path='cleaned-'+filename)