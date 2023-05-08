from flask import Flask, render_template, request, flash
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np



app = Flask(__name__)
app.secret_key = "verityfrom496"

#SVM Small Dataset
svmmodel = joblib.load('joblibs/smallsvmmodel.joblib')
svmcv = joblib.load('joblibs/smallsvmvec.joblib')

#SVM Large enron Dataset
lsvmmodel = joblib.load('joblibs/lsvmmodel.joblib')
lsvmcv = joblib.load('joblibs/lsvmvec.joblib')

#Naive Bayes Small Dataset
nbmodel = joblib.load('joblibs/smallnbmodel.joblib')
nbcv = joblib.load('joblibs/smallnbvec.joblib')

#Naive Bayes Large enron Dataset
lnbmodel = joblib.load('joblibs/lnbmodel.joblib')
lnbcv = joblib.load('joblibs/lnbvec.joblib')


@app.route('/index')
def home():
    return render_template("index.html")

@app.route('/research')
def emailChecker():
    flash("This email is...")
    flash('0%', 'per_message')
    return render_template("research.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/checkemail', methods=["POST", "GET"])
def checkemail():
    input_text = request.form['input']

    #'''
    # SVM Small Dataset
    svmit = svmcv.transform([input_text])
    svmnum = svmmodel.predict(svmit)[0]
    svmpercent = svmmodel.predict_proba(svmit)[0]
    svmresult = ""
    if svmnum == 1:
        svmresult = "spam"
    else:
        svmresult = "not spam"
    #'''

    #'''
    # SVM Large Dataset
    lsvmit = lsvmcv.transform([input_text])
    lsvmresult = lsvmmodel.predict(lsvmit)[0]
    lsvmpercent = lsvmmodel.predict_proba(lsvmit)[0]
    if lsvmresult == "ham":
        lsvmresult = "not spam"
    #'''

    #'''
    # NB Small Dataset
    nbit = nbcv.transform([input_text])
    nbnum = nbmodel.predict(nbit)
    nbpercent = nbmodel.predict_proba(nbit)
    nbresult = ""
    if nbnum == 1:
        nbresult = "spam"
    else:
        nbresult = "not spam"
    #'''

    #'''
    # NB Large Dataset
    lnbit = lnbcv.transform([input_text])
    lnbresult = lnbmodel.predict(lnbit)[0]
    lnbpercent = lnbmodel.predict_proba(lnbit)[0]
    if lnbresult == "ham":
        lnbresult = "not spam"
    #'''

    flash("SVM 1: This email is " + str(svmresult) + "  |  " + "Naive Bayes 1: This email is " + str(nbresult))
    flash("SVM 2: This email is " + str(lsvmresult) + "  |  " + "Naive Bayes 2: This email is " + str(lnbresult))

    #flash("SVM 1 percentages " + str(svmpercent))
    #flash("NB 1 percentages " + str(nbpercent))
    #flash("SVM 2 percentages " + str(lsvmpercent))
    #flash("NB 2 percentages " + str(lnbpercent))

    #flash("SVM 1 probability of spam is " + str(round(svmpercent[1], 2)))
    #flash(str(np.round(svmpercent[1], 2)*100) + "%", 'per_message')
    flash("SVM 1 probability of spam is ")
    flash("{:.0f}%".format(round(svmpercent[1] * 100, 2)), 'per_message')

    #flash("NB 1 probability of spam is  " + str(np.round(nbpercent[0][1], 2)))
    #flash(str(np.round(nbpercent[0][1], 2)*100) + "%", 'per_message')
    flash("NB 1 probability of spam is ")
    flash("{:.0f}%".format(round(nbpercent[0][1] * 100, 2)), 'per_message')

    #flash("SVM 2 probability of spam is  " + str(round(lsvmpercent[1], 2)))
    #flash(str(np.round(lsvmpercent[1], 2)*100) + "%", 'per_message')
    flash("SVM 2 probability of spam is ")
    flash("{:.0f}%".format(round(lsvmpercent[1] * 100, 2)), 'per_message')

    #flash("NB 2 probability of spam is  " + str(np.round(lnbpercent[1], 2)))
    #flash(str(np.round(lnbpercent[1], 2)*100) + "%", 'per_message')
    flash("NB 2 probability of spam is  ")
    flash("{:.0f}%".format(round(lnbpercent[1] * 100, 2)), 'per_message')


    return render_template("research.html")

if __name__ == '__main__':
    app.run()
