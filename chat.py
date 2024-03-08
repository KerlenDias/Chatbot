# Etapa 2: Instalação de módulos
print('Instalando módulos...')

import nltk
import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')
nltk.download('rslp')
nltk.download('stopwords')
print('Módulos instalados!')

# Etapa 3: Preparação dos dados em words, classes e documents
print('Preparação dos dados...')
stemmer = nltk.stem.RSLPStemmer()
words = []
classes = []
documents = []
for intent in intents['intents']:
for pattern in intent['patterns']:
w = nltk.word_tokenize(pattern.lower(), language=
'portuguese')
words.extend(w)
documents.append((w, intent['tag']))
if intent['tag'] not in classes:
classes.append(intent['tag'])
Chatbot com TensorFlow
def prepare_words(words):
# Se for string converte string para lista de palavras
if isinstance(words, str):
words = nltk.word_tokenize(words, language=
'portuguese')
stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords.append('?')
stopwords.append('!')
replace_words = [['vc', 'você'], ['ta', 'está'], ['estah', 'está']]
# Faz correção ortográfica
for replace_word in replace_words:
words = [replace_word[1] if replace_word[0]==w else w for w in words]
# Retira palavras que estão na lista de stopwords
words = [stemmer.stem(w.lower()) for w in words if w not in stopwords]
words = sorted(list(set(words)))
return words
words = prepare_words(words)
classes = sorted(list(set(classes)))
print('words:', words)
print('classes:', classes)
print('documents:', documents)
Chatbot com TensorFlow
# Etapa 4: Preparação dos dados de treinamento
print('Preparando dados de treinamento...')
training = []
output = []
output_empty = [0] * len(classes)
for doc in documents:
bag = []
pattern_words = doc[0]
pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
for w in words:
bag.append(1) if w in pattern_words else bag.append(0)
output_row = list(output_empty)
output_row[classes.index(doc[1])] = 1
training.append([bag, output_row])
random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print('train_x:', train_x[:5])
print('train_y:', train_y[:5])
Chatbot com TensorFlow
# Etapa 5: Criação da rede neural e treinamento
print('Criando e treinando rede neural...')
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation=
'softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net, tensorboard_dir=
'tflearn_logs')
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn’)
print(len(documents), "documentos")
print(len(classes), "classes", classes)
print(len(words), "raizes de palavras (stemming)", words)
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open("training_data", "wb"))
print('Rede Neural Treinada!')
Chatbot com TensorFlow
# Etapa 6: Funções bow(), classify() e response()
print("Iniciando teste de Rede Neural...")
data = pickle.load( open( "training_data"
,
"rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']
model.load('./model.tflearn')
def bow(sentence, words, show_details=False):
sentence_words = prepare_words(sentence)
bag = [0]*len(words)
for s in sentence_words:
for i,w in enumerate(words):
if w == s:
bag[i] = 1
if show_details:
print ("Encontrado na bag: %s" % w)
return(np.array(bag))
Chatbot com TensorFlow
def classify(sentence):
results = model.predict([bow(sentence, words)])[0]
# 0.25 é o limiar de erro (ERROR THRESHOLD)
results = [[i,r] for i,r in enumerate(results) if r>0.25]
results.sort(key=lambda x: x[1], reverse=True)
return_list = []
for r in results:
return_list.append((classes[r[0]], r[1]))
return return_list
def response(sentence):
results = classify(sentence)
if results:
while results:
for i in intents['intents']:
if i['tag'] == results[0][0]:
return print(random.choice(i['responses']))
results.pop(0)
print('Exemplos:')
print('Classificação:', classify('vc namora?'))
print('Resposta de "vc namora?":')
response('vc namora?')
Chatbot com TensorFlow
# Etapa 7: Execução da rede neural
while True:
pergunta = input('Digite algo para a rede neural: ')
print('Classificação:', classify(pergunta))
if classify(pergunta)[0][0] == 'despedir':
response(pergunta)
break
if classify(pergunta)[0][1] > 0.50:
response(pergunta)
else:
print('Desculpa, não entendi!')
print('Fim do chatbot!')