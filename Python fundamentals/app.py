from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

acao1 = [1, 0, 1]
acao2 = [0, 1, 0]
acao3 = [1, 1, 1]
acao4 = [0, 0, 1]
acao5 = [1, 1, 0]
acao6 = [0, 1, 1]

dados_treino = [acao1, acao2, acao3, acao4, acao5, acao6]
rotulos_treino = [1, 1, 1, 0, 0, 0]

modelo = LinearSVC()
modelo.fit(dados_treino, rotulos_treino)

teste1 = [1, 0, 0]
teste2 = [0, 1, 1]
teste3 = [1, 1, 0]

dados_teste = [teste1, teste2, teste3]
rotulos_teste = [1, 0, 1]

previsoes = modelo.predict(dados_teste)

acuracia = accuracy_score(rotulos_teste, previsoes)

print("Taxa de acerto %.2f%%" % (acuracia * 100))