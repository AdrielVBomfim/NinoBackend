from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import nltk
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import datastore

'''phrases = ["Could you send me a picture of you",
           "I love you",
           "I love you boy",
           "Why dont you give me your phone number",
           "You are beautiful",
           "You are beautiful girl",
           "You are pretty little girl",
           "How old are you",
           "where do you live"]'''

#Trecho de código para lógica das rotas da API

class TextSimilarity:
    def __init__(self, init=1):
        if (init == 1):
            self.statements = [
                'I love you',
                'Could you send me some photos?',
                'I think you so kind',
                'Can we have a meet at shopping?',
                'Would you like to have a meet out of here?',
                'I have some toys that would you like it'
            ]
        else:
            self.statements = []   # Na API os statements serao puxadas de um banco de dados NoSQL

            query = datastore_client.query(kind='suspSentence')
            results = list(query.fetch())

            for result in results:
                self.statements.append(result['sentence'])

    def TF(self, frase):
        palavras = nltk.word_tokenize(frase.lower())
        freq = nltk.FreqDist(palavras)
        dicionario = {}
        for chave in freq.keys():
            norm = freq[chave] / float(len(palavras))
            dicionario[chave] = norm
        return dicionario

    def IDF(self):
        def idf(Num_Documentos, Num_Documentos_com_Palavra):
            return 1.0 + math.log(Num_Documentos / Num_Documentos_com_Palavra)

        num_Documentos = len(self.statements)
        unique_palavra = {}
        idf_Valores = {}
        for frase in self.statements:
            for palavra in nltk.word_tokenize(frase.lower()):
                if palavra not in unique_palavra:
                    unique_palavra[palavra] = 1
                else:
                    unique_palavra[palavra] += 1
        for palavra in unique_palavra:
            idf_Valores[palavra] = idf(num_Documentos, unique_palavra[palavra])
        return idf_Valores

    def TF_IDF(self, pilha):
        palavras = nltk.word_tokenize(pilha.lower())
        idf = self.IDF()
        vetores = {}
        for frase in self.statements:
            tf = self.TF(frase)
            for palavra in palavras:
                tfv = tf[palavra] if palavra in tf else 0.0
                idfv = idf[palavra] if palavra in idf else 0.0
                mult = tfv * idfv
                if palavra not in vetores:
                    vetores[palavra] = []
                vetores[palavra].append(mult)
        return vetores

    def mostrar_Vetores(self, vetores):
        print(self.statements)
        for palavra in vetores:
            print("{}->{}".format(palavra, vetores[palavra]))

    # Funcao modificada para indicar a proximidade, do ponto de vista de similarida
    # de do documento em questao para com o resto. Logo, no processo de conta deve-
    # se retirar o elemento diagonal que sempre resultará na maxima semelhança e, a
    # partir daí contar novamente qual o documento mais próximo após a exclusão da
    # diagonal
    def dist_cosseno(self, hasInput):
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(self.statements)
        debug = ""

        for j in range(1, len(self.statements) + 1):
            i = j - 1
            max_value = 0.0
            max_sim = 0
            debug = debug + "           Similarity of the Document {} with the others".format(j)
            similaridade = cosine_similarity(matrix[i:j], matrix)
            for kk in range(len(similaridade[0])):
                if j - 1 != kk and similaridade[0][kk] > max_value:
                    max_value = similaridade[0][kk]
                    max_sim = kk + 1

            debug = debug + str(similaridade[0])
            debug = debug + " Document {} more similar to Document {}".format(j, max_sim)

            if hasInput and j == len(self.statements):
                temp = similaridade[0].tolist()
                temp.pop(len(similaridade[0]) - 1)
                return max(temp)

        return debug

    def phrasePrediction(self, inputPhrase):
        if inputPhrase.upper() not in [statement.upper() for statement in self.statements]:
            self.statements.append(inputPhrase)
            probability = self.dist_cosseno(True)
            self.statements.remove(inputPhrase)
            return round(probability * 100, 2)
        else:
            return 100.00

    def addSuspiciousStatement(self, phraseToBeAdded):
        if phraseToBeAdded not in self.statements:
            self.statements.append(phraseToBeAdded)     # Na API a frase será adicionada no banco de dados NoSQL

    def demo(self):
        pilha_ent = self.statements[0]
        vetores = self.TF_IDF(pilha_ent)
        self.mostrar_Vetores(vetores)
        self.dist_cosseno(False)

datastore_client = datastore.Client()

#Trecho de código para definição estrutural da API

app = Flask(__name__)
CORS(app)

api = Api()
api.init_app(app, version='0.1', title='NinoAPI', description='This is the API of the Nino Project. It gives access to some of the functionalities of our solution to predict cybercrimes involving children.'
                                                             '\n\nPlease refer to the documentation below for more information about what our solution can do.')

sentence = api.model('Suspicious_Sentence', {
    'phrase': fields.String
})

probability = api.model('Pedophile_Probability', {
    'probability': fields.Float
})

@api.route('/prediction', doc={"description": "Returns the probability for the input phrase be from a pedophile."})
@api.doc(responses={200: """Success""", 400: """Phrase to be analised was not sent"""})
class prediction(Resource):
    @api.doc(body=sentence, model='Pedophile_Probability')
    def post(self):
        A = TextSimilarity(init=2)
        req_data = request.get_json()

        if 'phrase' in req_data:
            probability = A.phrasePrediction(req_data['phrase'])
        else:
            return 'Phrase to be analised was not sent', 400

        return {"probability": probability}

@api.route('/sentences', doc={"description": "Returns all the sentences which are stored in the project associated Database and used as reference for our solution to predict cybercrimes."})
@api.doc(responses={200:  """Success\n\nReturn model: ["I love you", "I like the way you play", "Can we talk somewhere else?", ...]"""})
class showSentences(Resource):
    @api.doc()
    def get(self):
        A = TextSimilarity(init=2)
        return jsonify(getattr(A, 'statements'))

@api.route('/comparison', doc={"description": "Returns a report showing the level of similarity of all phrases stored in the project associated Database with themselves."})
@api.doc(responses={200: """Success\n\nReturn model: "Similarity of the Document {x} with the others[[1.         0.02323458 0.24290615]] Document {y} more similar to Document ...\""""})
class similarityAll(Resource):
    def get(self):
        A = TextSimilarity(init=2)
        debug = A.dist_cosseno(False)
        return jsonify(debug)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, processes=10)