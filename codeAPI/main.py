from flask import Flask
from flasgger import Swagger
from flask_restful import Api, Resource
import pandas as pd
import joblib
import spacy
import en_core_web_sm
import cleaning as cl

app = Flask(__name__)
api = Api(app)

template = {
  "swagger": "2.0",
  "info": {
    "title": "Générateur de Tags pour les questions du site Stack Overflow",
    "description": "API de prédiction de Tags : prétraitement NLP et prédictions avec la Regression logistique multi-labels (avec 42,84% des valeurs non prédites)", 
  }
}

swagger = Swagger(app, template=template)

# Load pre-trained models
model_path = "models/"
vectorizer = joblib.load(model_path + "tfidf_vectorizer.pkl", 'r')
multilabel_binarizer = joblib.load(model_path + "multilabel_binarizer.pkl", 'r')
model = joblib.load(model_path + "logit_nlp_model.pkl", 'r')

class Tags(Resource):
    def get(self, question):
        """
       Pour tester : Allez sur le site Stack Overflow, Copiez une question, collez la sur le champ specifié, exécutez le modèle et récupérez le résultat.
       ---
       parameters:
         - in: path
           name: question
           type: string
           required: true
       responses:
         '200':
           description: Predicted list of tags
           content:
               application/json:
                   schema:
                       type: object
                       properties:
                           Predicted_Tags:
                               type: string
                               description: List of predicted tags 
        """
        # Clean the question sent
        nlp = en_core_web_sm.load()
        #nlp = spacy.load()
        pos_list = ["NOUN","PROPN"]
        rawtext = question
        cleaned_question = cl.text_cleaner(rawtext, nlp, pos_list, "english")
        
        # Apply saved trained TfidfVectorizer
        X_tfidf = vectorizer.transform([cleaned_question])
        
        # Perform prediction
        predict = model.predict(X_tfidf)
        
        # Inverse multilabel binarizer
        tags_predict = multilabel_binarizer.inverse_transform(predict)
              
        # Results
        results = {}
        results['Predicted_Tags'] = tags_predict  
        return results, 200

api.add_resource(Tags, '/tags/<question>')

if __name__ == "__main__":
	app.run()