import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_gs   = pd.read_csv("gender_submission.csv")

# print("Linhas x colunas - gender_submission")
# print(df_gs.shape)

# print("########################")
# print("Linhas x colunas - train")
# print(df_train.shape)

# print("########################")
# print("Linhas x colunas - test")
# print(df_test.shape)

df_train.columns=['PassageiroId','Sobrevivente','ClasseTicket','Nome','Sexo','Idade','irmaosConjuges','paisCriancas','ticket','tarifa','Cabine','Embarque']

df_train['Sexo'] = pd.factorize(df_train['Sexo'])[0]
df_train['Idade'] = df_train['Idade'].fillna(0)
df_train['Embarque'] = pd.factorize(df_train['Embarque'])[0]

## Verificando em quais campos possuem valores nulos
# print('Categorias de Sobrevivente:' ,df_train.Sobrevivente.unique())
# print('Categorias de Classe Ticket:',df_train.ClasseTicket.unique())
# print('Categorias de Sexo:'         ,df_train.Sexo.unique())
# print('Categorias de Idade:'        ,df_train.Idade.unique())
# print('Categorias de irmaosConjuges:',df_train.irmaosConjuges.unique())
# print('Categorias de paisCriancas:' ,df_train.paisCriancas.unique())
# print('Categorias de tarifa:'       ,df_train.tarifa.unique())
# print('Categorias de Embarque:'     ,df_train.Embarque.unique())

x = df_train[['ClasseTicket','Sexo','Idade','irmaosConjuges','paisCriancas','tarifa','Embarque']]
y = df_train['Sobrevivente']

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y,
                                                         random_state = 42, test_size = 0.4,
                                                         stratify = y)

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)
print(treino_x.shape,treino_y.shape,teste_x.shape,teste_y.shape)
acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)