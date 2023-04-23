from flask import Flask, render_template, redirect, url_for, request, session, flash, escape, current_app as apps, make_response, Response,send_file
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from hexamer_features import fileoutput
from CNN_Phylum import phylum_model
from CNN_Class import class_model
from CNN_Order import order_model
from CNN_Family import family_model
from CNN_Genus import genus_model
from CNN_Species import species_model

app=Flask(__name__)
from subprocess import Popen, PIPE


@app.route('/', methods = ['GET','POST'])
def homed():
    hjjkhkl=0
    return render_template('pg_home.html')
@app.route('/upload_page_feat', methods = ['GET','POST'])
def upload_page_feat():
    hjjkhkl =0
    return render_template('feature.html')
@app.route('/upload_page_class', methods = ['GET','POST'])
def upload_page_class():
    hjjkhkl='hello'
    return render_template('classification.html',msg=hjjkhkl)
@app.route('/algo', methods = ['GET','POST'])
def algo():
    hjjkhkl='hello'
    return render_template('algorithm.html',msg=hjjkhkl)
@app.route('/cont', methods = ['GET','POST'])
def cont():
    hjjkhkl='hello'
    return render_template('contact.html',msg=hjjkhkl)
@app.route('/upload_feat', methods = ['GET','POST'])
def upload_feat():
    dft=request.files['csvfile']
    num=request.form['number']
    filename=dft.filename
    if filename.endswith(".fasta"):
        dft.save(filename)
    fileoutput(filename,num)

    from subprocess import Popen, PIPE

    return render_template('uploaded_feat.html',msg=filename)
#####################################
@app.route('/upload_class', methods = ['GET','POST'])
def upload_class():
    dft = request.files.get('csvfile')
    num = request.form['number']
    filename = 'features.csv'   # Set filename as 'features.csv'
    filepath = os.path.join('Results', 'classification_feature',filename)
    dft.save(filepath)
    if (int (num)==1):
        all_taxa_model(pd.read_csv(filepath))
        return render_template('uploaded_all_taxa.html')
    elif (int(num)==2):
        phylum_model(pd.read_csv(filepath))
        return render_template('uploaded_phylum.html')
    elif (int(num)==3):
        class_model(pd.read_csv(filepath))
        return render_template('uploaded_class.html')
    elif (int(num)==4):
        order_model(pd.read_csv(filepath))
        return render_template('uploaded_order.html')
    elif (int(num)==5):
        family_model(pd.read_csv(filepath))
        return render_template('uploaded_family.html')
    elif (int(num)==6):
        genus_model(pd.read_csv(filepath))
        return render_template('uploaded_genus.html')
    else:
        species_model(pd.read_csv(filepath))
        return render_template('uploaded_species.html')


###################################################
@app.route('/download_hexamer_features', methods=['GET', 'POST'])
def download_hexamer_features():
    path=os.path.join(os.getcwd(), 'Results',
                              'feature_extraction_results', 'features.csv')
    #path = "/Examples.pdf"
    return send_file(path, as_attachment=True)

### All taxa outputs
@app.route('/download_all_labels',methods = ['GET','POST'])
def download_all_labels():
    path=os.path.join(os.getcwd(), 'All_Taxa_Classification_Results','Predicted_Labels.csv')
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)
@app.route('/download_phylum_proba',methods = ['GET','POST'])
def download_phylum_proba():
    path=os.path.join(os.getcwd(), 'All_Taxa_Classification_Results','PredictionProbabilities_phylum.csv')
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)
@app.route('/download_class_proba',methods = ['GET','POST'])
def download_class_proba():
    path=os.path.join(os.getcwd(), 'All_Taxa_Classification_Results','PredictionProbabilities_class.csv')
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)
@app.route('/download_order_proba',methods = ['GET','POST'])
def download_order_proba():
    path=os.path.join(os.getcwd(), 'All_Taxa_Classification_Results','PredictionProbabilities_order.csv')
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)
@app.route('/download_family_proba',methods = ['GET','POST'])
def download_family_proba():
    path=os.path.join(os.getcwd(), 'All_Taxa_Classification_Results','PredictionProbabilities_family.csv')
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)
@app.route('/download_genus_proba',methods = ['GET','POST'])
def download_genus_proba():
    path=os.path.join(os.getcwd(), 'All_Taxa_Classification_Results','PredictionProbabilities_genus.csv')
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)
@app.route('/download_species_proba',methods = ['GET','POST'])
def download_species_proba():
    path=os.path.join(os.getcwd(), 'All_Taxa_Classification_Results','PredictionProbabilities_species.csv')
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)

### Phylum outputs
@app.route('/download_phylum_labels',methods = ['GET','POST'])
def download_phylum_labels():
    path=os.path.join(os.getcwd(), 'Phylum_Classification_Results','Predicted_Phyla.csv')
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)
@app.route('/download_phylum_probabilities',methods = ['GET','POST'])
def download_phylum_probabilities():
    path=os.path.join(os.getcwd(), 'Phylum_Classification_Results','PredictionProbabilities_phylum.csv')
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)

### Class outputs
@app.route('/download_class_labels',methods = ['GET','POST'])
def download_class_labels():
    path=os.path.join(os.getcwd(), 'Class_Classification_Results','Predicted_Classes.csv')
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)
@app.route('/download_class_probabilities',methods = ['GET','POST'])
def download_class_probabilities():
    path=os.path.join(os.getcwd(), 'Class_Classification_Results','PredictionProbabilities_class.csv')      
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)

### Order outputs
@app.route('/download_order_labels',methods = ['GET','POST'])
def download_order_labels():
    path=os.path.join(os.getcwd(), 'Order_Classification_Results','Predicted_Orders.csv')      
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)
@app.route('/download_order_probabilities',methods = ['GET','POST'])
def download_order_probabilities():
    path=os.path.join(os.getcwd(), 'Order_Classification_Results','PredictionProbabilities_order.csv')      
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)

### Family outputs
@app.route('/download_family_labels',methods = ['GET','POST'])
def download_family_labels():
    path=os.path.join(os.getcwd(), 'Family_Classification_Results','Predicted_Families.csv')      
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)
@app.route('/download_family_probabilities',methods = ['GET','POST'])
def download_family_probabilities():
    path=os.path.join(os.getcwd(), 'Family_Classification_Results','PredictionProbabilities_family.csv')      
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)

### Genus outputs
@app.route('/download_genus_labels',methods = ['GET','POST'])
def download_genus_labels():
    path=os.path.join(os.getcwd(), 'Genus_Classification_Results','Predicted_Genera.csv')      
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)
@app.route('/download_genus_probabilities',methods = ['GET','POST'])
def download_genus_probabilities():
    path=os.path.join(os.getcwd(), 'Genus_Classification_Results','PredictionProbabilities_genus.csv') 
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)

### Species outputs
@app.route('/download_species_labels',methods = ['GET','POST'])
def download_species_labels():
    path= os.path.join(os.getcwd(), 'Species_Classification_Results', 'Predicted_Species.csv')
    return send_file(path, as_attachment=True)
@app.route('/download_species_probabilities',methods = ['GET','POST'])
def download_species_probabilities():
    path= os.path.join(os.getcwd(), 'Species_Classification_Results', 'PredictionProbabilities_species.csv')
    # path = "/Examples.pdf"
    return send_file(path, as_attachment=True)



if __name__ == '__main__':
    app.run()
