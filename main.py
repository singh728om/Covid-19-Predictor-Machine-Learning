from flask import Flask, render_template, request

app = Flask(__name__)
import pickle

file = open('model.pkl', 'rb')

clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET","POST"])
def Demo_Page():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        bodyPain = int(myDict['bodyPain'])
        age = int(myDict['age'])
        runnyNose = int(myDict['runnyNose'])
        Moderate_severe_cough = int(myDict['Moderate_severe_cough'])
        Dry_Cough = int(myDict['Dry_Cough'])
        Gender = int(myDict['Gender'])
        Sore_throat = int(myDict['Sore_throat'])
        Weakness = int(myDict['Weakness'])
        Change_in_Appetite = int(myDict['Change_in_Appetite'])
        Feeling_breathless = int(myDict['Feeling_breathless'])
        close_contact = int(myDict['close_contact'])
        Diabetes = int(myDict['Diabetes'])
        heart_dis = int(myDict['heart_dis'])
        progressin48hr = int(myDict['progressin48hr'])
        kidneydis = int(myDict['kidneydis'])

    #inference_input
        inputFeature =[fever,bodyPain,age,runnyNose,Moderate_severe_cough,Dry_Cough,Gender,Sore_throat,Weakness,Change_in_Appetite,Feeling_breathless,close_contact,Diabetes,heart_dis,progressin48hr,kidneydis]
        infProb = clf.predict_proba([inputFeature])[0][1]
        print(infProb)
        return render_template('show.html', inf=round(infProb*100))
        #return render_template('show.html', inf=infProb)
    return render_template('index.html')
    #return 'Probability of Corona Virus: ' + str(infProb)

if __name__ == "__main__":
    app.run(debug=True)