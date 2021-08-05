#!/usr/bin/env python
# coding: utf-8

# As doenças cardiovasculares hoje são a maior causadora de mortes no mundo, estimado que 31% das mortes no mundo estão relacionadas a doenças cardiovasculares. Essas doenças causam diretamente a falha do coração que levam a morte do individuo.
# 
# A maioria das doenças cardiovasculares podem ser evitadas com pequenos cuidados, porém é dificil saber se algum individuo irá possuir uma doença cardiovascular, dessa forma Machine Learning é de grande utilidade para auxiliar na prevenção dessas doenças.
# 
# O objetivo desse projeto é a previsão se o paciente irá ser levado a uma falha cardiovascular pelo desenvolvimento de alguma doença.
# 
#      -- Objetivos
#  
#       É esperado um Recall minimo de 75% para a classe 0 (DEATH_EVENT não ocorreu). Já para a classe 1 (DEATH_EVENT ocorreu) é esperado um Recall de 85%. 
#      
#       A escolha dos objetivos é baseado em algumas analises iniciais, sabendo que o nosso dataset não possui muitas observa-ções e esta desbalanceado. Dessa forma irá precisar de uma modelagem mais cuidadosa para atingir um modelo ideal. 
#      
#       Também foi escolhido que iremos priorizar que o algoritmo acerte quem irá MORRER. Dessa forma poderia ser iniciado um tratamento para o paciente antecipadamente.
#      
# Dataset: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

# ## Variaveis
# 
# Explicadação dada pelo autor no tópico: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/discussion/193109
# 
# 
# |     Feature                    | Explanation                                     |  Measurement     | Range               |   |
# |:------------------------------:|:-----------------------------------------------:|:----------------:|:-------------------:|---|
# | Age                            | Age of the patient                              | Years            | [40,…, 95]          |   |
# | Anaemia                        | Decrease of red blood cells or hemoglobin       | Boolean          | 0, 1                |   |
# | High blood pressure            | If a patient has hypertension                   | Boolean          | 0, 1                |   |
# | Creatinine phosphokinase (CPK) | Level of the CPK enzyme in the blood            | mcg/L            | [23,…, 7861]        |   |
# | Diabetes                       | If the patient has diabetes                     | Boolean          | 0, 1                |   |
# | Ejection fraction              | Percentage of blood leaving                     | Percentage       | [14,…, 80]          |   |
# | Sex                            | Woman or man                                    | Binary           | 0, 1                |   |
# | Platelets                      | Platelets in the blood                          | kiloplatelets/mL | [25.01,…, 850.00]   |   |
# | Serum creatinine               | Level of creatinine in the blood                | mg/dL            | [0.50,…, 9.40]      |   |
# | Serum sodium                   | Level of sodium in the blood                    | mEq/L            | [114,…, 148]        |   |
# | Smoking                        | If the patient smokes                           | Boolean          | 0, 1                |   |
# | Time                           | Follow-up period                                | Days             | [4,…,285]           |   |
# | (target) death event           | If the patient died during the follow-up period | Boolean          | 0, 1                |   |
# 
#         * Sex - Gender of patient Male = 1, Female =0
#         * Diabetes - 0 = No, 1 = Yes
#         * Anaemia - 0 = No, 1 = Yes
#         * High_blood_pressure - 0 = No, 1 = Yes
#         * Smoking - 0 = No, 1 = Yes
#         * DEATH_EVENT - 0 = No, 1 = Yes

# ## Imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import imblearn
import os
import pickle
import time
import shap

from pathlib import Path
from scipy import stats
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from pyclustertend import hopkins
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from math import ceil
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from xgboost import plot_importance
from lifelines import CoxPHFitter
from warnings import simplefilter


# ## Ambiente

# In[2]:


simplefilter(action='ignore', category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_theme()


# In[3]:


seed_ = 194
np.random.seed(seed_)


# ## Coleta dos Dados

# In[4]:


heart_original = pd.read_csv('Data/heart.csv', sep = ',')


# # 1. Analise Exploratoria

# In[5]:


heart_original.head()


# É verificado que possuimos apenas 299 registros, o que significa que possuimos poucos dados para trabalhar. Assim, temos que tomar cuidado com a manipulação nos dados e tentar evitar qualquer remoção de registro. É perceptivel que o dataset é composto por pessoas acima de 40 anos, e com uma média de 60 anos, levando a uma população mais velha.

# In[6]:


heart_original.describe()


# É verificado que a variavel 'Age' esta classificada como float, o que pode ser um erro e iremos converter para inteiro. Já em relação as outras variaveis não tem nada errado, algumas variaveis que poderiam ser boolean estão como inteiro, mas isso não causa nenhum problema para o algoritmo.

# In[7]:


heart_original.dtypes


# In[8]:


heart = heart_original.copy()


# In[9]:


heart['age'] = heart['age'].astype('int64')


# In[10]:


heart.dtypes


# Também é verificado que nenhuma informação é perdida com essa transformação.

# In[11]:


heart.describe()


# In[12]:


# Verificando se possui valores missing
print(heart.isna().sum())


# In[13]:


# Verificando valores unicos
print(heart.nunique())


# In[14]:


# Verificando valores duplicados
print(sum(heart.duplicated()))


# Para uma melhor compreensão iremos separar o dataset em variaveis categoricas e continuas, assim podemos utilizar de tecnicas especificas para cada tipo.

# In[15]:


# Lista de variaveis de cada tipo
continuas = []
categoricas = []

for c in heart.columns[:-1]:
    if heart.nunique()[c] > 5:
        continuas.append(c)
    else:
        categoricas.append(c)


# In[16]:


continuas


# In[17]:


categoricas


# In[18]:


heart[continuas].head()


# In[19]:


heart[categoricas].head()


# Verificando o boxplot abaixo das variaveis continuas, é notavel que a variavel 'creatinine_phosphokinase', 'serum_creatinine' e 'platelets' possuem um auto numero de outliers, principalmente as duas primeiras. Esses outliers estão concentrados acima do Q3.

# In[20]:


# Plot para variaveis continuas

fig = plt.figure(figsize = (12, 8))

for i, col in enumerate(continuas):
    plt.subplot(3, 3, i + 1)
    heart.boxplot(col)
    plt.tight_layout()


# Para uma melhor visualização iremos tentar transformar as variaveis continuas para log.

# In[21]:


heart[continuas] = np.log1p(1 + heart[continuas])


# A transformação nos levou a alguns insights, como que a variavel 'time' pode possuir outliers se convertido em log, já outras variaveis como 'creatinine_phosphokinase' passam a não ter presença significativa de outliers.

# In[22]:


# Plot para variaveis continuas

fig = plt.figure(figsize = (12, 8))

for i, col in enumerate(continuas):
    plt.subplot(3, 3, i + 1)
    heart.boxplot(col)
    plt.tight_layout()


# In[23]:


heart = heart_original.copy()


# É notado que as variaveis continuas possuem uma tendência para inversamente proporcional, porém essa tendencia é fraca, mostrando valores quase que nulos para a correlação. É notado que a variavel 'creatinine_phosphokinase' e 'platelets' não aparentam possuir correlação com a variavel target. 
# 
# É perceptivel que Time, esta altamente correlacionado com a variavel target, pois é uma variavel que não pode pertencer ao dataset, visto que indica o tempo em meses até a falha cardiovascular do paciente depois da consulta ao medico, visto que essa variavel não seria capturada em um cenario real deve ser removida do dataset.

# In[24]:


# Mapa de calor das variaveis continuas
continuas_temp = continuas.copy()
continuas_temp.append('DEATH_EVENT')

plt.figure(figsize = (12, 12))
sns.heatmap(heart[continuas_temp].corr(method = 'pearson'), annot = True, square = True)
plt.show()


# In[25]:


# Countplot para variaveis categóricas

fig = plt.figure()
fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
fig.set_figheight(7)
fig.set_figwidth(10)

for i, col in enumerate(categoricas):
    ax = fig.add_subplot(ceil(len(categoricas) / 3), 3, i + 1)
    sns.countplot(x = heart[col])
    
plt.tight_layout()
plt.show()


# É verificado que o nosso dataset possui um desbalaneamento notavel, visto que aproximadamente 66% das variaveis é da classe 0 (não ocorreu falha cardiovascular) e 33% da classe 1 (ocorreu falha cardiovascular).

# In[26]:


# Countplot da variavel target
sns.countplot(x = heart['DEATH_EVENT'])
plt.show()


# In[27]:


print(Counter(heart['DEATH_EVENT']))


# Para alguns insights iremos comparar a relação por um stacked barplot entre as variaveis categoricas e target. Para os graficos abaixo a grande surpresa fica no ultimo, 'smoking' onde é percebido que os não fumantes, classe 0, morrem mais que os fumantes, classe 1. Claro, isso é uma analise de somente uma métrica, para se ter certeza teriamos que analisar mais metricas.

# In[28]:


for col in categoricas:
    pd.crosstab(heart[col], heart['DEATH_EVENT']).plot(kind = 'bar',
                                                       stacked = True,
                                                       figsize = (15, 5),
                                                       color = ['green', 'red'])


# ## 1.1 Distribuição dos Dados

# Iremos verificar se os nossos dados possuem uma distribuição Gaussiana ou não. Dessa forma iremos entender quais metodos estatisticos utilizar.
# Distribuições Gaussianas utilizam de métodos estatisticos paramétricos. Já o contrário utiliza de métodos estatisticos não paramétricos. É importante entender qual método utilizar para não termos uma vissão errada sobre os dados.

# In[29]:


def quantil_quantil_teste(data, columns):
    
    for col in columns:
        print(col)
        qqplot(data[col], line = 's')
        plt.show()


# Analisando as variaveis abaixo:
# 
# Age: Aparenta ser uma variavel com o comportamento Gaussiano, possui alguns pontos fora da linha mas nada muito grave.
# 
# creatinine_phosphokinase: Não apresenta comportamento Gaussiano, possui muitos pontos constantes e esta fora da linha de dados Gaussianos.
# 
# ejection_fraction: Possui um leve comportamento Gaussiano, os pontos tendem a seguir a reta, porém muitos pontos estão saindo da linha.
# 
# platelets: Possui forte comportamento Gaussiano, possui alguns pontos fora da linha mas nada grave.
# 
# serum_creatinine: Não apresenta comportamento Gaussiano, possui muitos pontos constantes e esta fora da linha de dados Gaussianos. Apesar disso parece seguir levemente a linha.
# 
# serum_sodium: Possui forte comportamento Gaussiano, possui alguns pontos fora da linha mas nada grave.
# 
# time: Apresenta fortes indicios de comportamento Gaussiano, porém os dados estão constantemente cruzando a reta.

# In[30]:


quantil_quantil_teste(heart_original, continuas)


# Abaixo iremos utilizar 3 metodos estatisticos baseado em Hipotese para avalidar os nossos dados, onde:
# 
# p <= alfa: Rejeita H0, dados não são Gaussianos.
# 
# p > alfa: Falha ao rejeitar H0, dados são Gaussianos.

# In[31]:


def testes_gaussianos(data, columns, teste):
    
    for i, col in enumerate(columns):
        print('Teste para a variavel', col)
        alpha = 0.05
        
        if teste == 'shapiro':
            stat, p = shapiro(data[col])
        elif teste == 'normal':
            stat, p = normaltest(data[col])           
        elif teste == 'anderson':
            resultado = anderson(data[col])
            print('Stats: %.4f' % resultado.statistic)
            
            for j in range(len(resultado.critical_values)):
                sl, cv = resultado.significance_level[j], resultado.critical_values[j]
                
                if resultado.statistic < cv:
                    print('Significancia = %.4f, Valor Critico = %.4f, os dados parecem Gaussianos. Falha ao rejeitar H0.' % (sl, cv))
                else:
                    print('Significancia = %.4f, Valor Critico = %.4f, os dados não parecem Gaussianos. H0 rejeitado.' % (sl, cv))
            
        if teste != 'anderson':         
            print('Stat = ', round(stat, 4))
            print('p-value = ', round(p, 4))
            #print('Stats = %4.f, p = %4.f' % (stat, p))

            if p > alpha:
                print('Os dados parecem Gaussianos. Falha ao rejeitar H0.')
            else:
                print('Os dados não parecem Gaussianos. H0 rejeitado.')
            
        print('\n')


# ### 1.1.1 Tesde de Shapiro Wilk

# O teste de Shapiro Wilk é ideal para avaliar amostras ou conjuntos menores de dados, avaliando a probabilidade de que os dados tenham sido extraidos de uma distribuição Gaussiana.

# In[32]:


testes_gaussianos(heart, continuas, teste = 'shapiro')


# ### 1.1.2 Teste normal de D'Agostino

# O teste Normal de D'Agostino avalia se os dados são Gaussianos utilizando estatisticas resumidas como: Curtose e Skew.

# In[33]:


testes_gaussianos(heart, continuas, teste = 'normal')


# ## 1.2 Tabela de Contigencia

# In[34]:


def crosstab_column(data, col, target, percentage = True):
    res = pd.crosstab(data[col], data[target], margins = True)
    
    if percentage:
        res = pd.crosstab(data[col], data[target], margins = True, normalize = 'index').round(4) * 100
    
    return res


# Analisando a tabela de contigência abaixo não é possível afirmar se uma variavel categorica possui relação para a classificação da variavel target. Para confirmarmos as nossas hipoteses, iremos seguir para a Correlação de Spearman e Qui-Quadrado.

# In[35]:


for col in categoricas:
    print(crosstab_column(heart, col, 'DEATH_EVENT'), end = '\n\n\n')


# ## 1.3 Correlação de Spearman

# In[36]:


def coefSpearman(data, col, target):    
    for c in col:
        coeficiente, p_valor = stats.spearmanr(data[c], data[target])
        print("Correlação de Spearman entre a variavel", target, "e a variavel continua", c, ": {:0.4}".format(coeficiente))


# Analisando a Correlação de Spearman abaixo as variaveis Categoricas possuem correlação semelhante com a de Pearson, porém um pouco menor. Isso significa que os nossos dados possuem uma correlação mais constante, entretando os nossos dados também possuem variaveis que não aumentam de forma constante.

# In[37]:


coefSpearman(heart, continuas, 'DEATH_EVENT')


# ## 1.4 QUI-QUADRADO

# In[38]:


def qui2(data, col, target):
    for c in col:
        cross = pd.crosstab(data[c], data[target])
        chi2, p, dof, exp = stats.chi2_contingency(cross)
        print("Qui-quadrado entre a variavel", target, "e a variavel categorica", c, ": {:0.4}".format(chi2))
        print("Apresentando um p-value de: {:0.4}".format(p), end = '\n\n')


# Comparando o p-value para as variaveis categorigas nós não conseguimos rejeitar a hipotese nula, concluindo que as nossas variaveis não estão correlacionadas com a target.

# In[39]:


qui2(heart, categoricas, 'DEATH_EVENT')


# ## 1.5 Cox Proportional Hazards

# Iremos utilizar o algoritmo de Cox Proportional Hazards, já que é famoso por associar as variaveis preditoras corretas para aumentar a chance de sobrevivencia de um individuo. Esse algoritmo é amplamente utilizado para dados médicos, sendo assim irá ser traçado a chance de sobrevivencia pelo tempo.
# 
# O dataset utilizado possui uma variavel 'time', essa que mede o tempo da ultima consulta do paciente até a falha cardiovascular. Já que essa variavel não irá ser utilizada no dataset final, visto que não iremos saber quando o paciente irá morrer, iremos utiliza-la antes para associar as variaveis. 

# In[40]:


cph = CoxPHFitter()


# In[41]:


cph.fit(heart, event_col = 'DEATH_EVENT', duration_col = 'time')


# O modelo nos indica que as variaveis 'high_blood_pressure', 'anaemia', 'serum_creatinine', 'diabetes', 'smoking' e 'age'. Apresentam forte confiança para indicar os riscos de falha cardiovascular.

# In[42]:


cph.plot()


# Abaixo possuimos um resumo estatistico apresentado pelo modelo:
# 
# Valores de 'p' onde (p < 0.05) indicam que as outras colunas possuem uma avaliação confiavel e possuem uma boa significancia para o modelo.
# 
# Altos valores de 'coef' indicam que o aumento dessa variavel leva ao aumento do risco de falha cardiovascular.  É perceptivel esses valores para a variavel 'anaemia', 'high_blood_pressure' e 'serum_creatinine'.
# 
# O valor de 'coef', 'exp(coef)', 'exp(coef) lower 95%' e 'exp(coef) upper 95%' estão relacionados do maior valor para o menor. Por exemplo 'anaemia' é uma variavel categorica com valor 1 e 0. Então estamos relacionando a chance de ter anaemia (1) com a chance de não ter (0).
# 
# O que significa que para a variavel 'coef' o seu aumento, valor 1, aumenta o risco de falha cardiovascular.
# 
# O atributo 'exp(coef)' indica a relação de riscos covariaods a variavel. De forma que o aumento daquela variavel aponta um risco Y em um fator X, X = 'exp(coef)'. Por exemplo a variavel 'Anaemia' tem que o seu aumento é prejudicial de acordo com 'coef', e temos em 'exp(coef)' que o seu aumento é prejudicial em um fator de 58% (1 - 1.58), por outra lado a sua diminuição é 42% benéfica (1 - 0.58).
# 
# Os atributos 'exp(coef) lower 95%' e 'exp(coef) upper 95%' indicam  os seus intervalos de confiança.
# 
# Já para os atributos globais, possuimos uma 'concordance' alta, o que é bom. Seguindo para 'Partial AIC' percebe-se que o modelo utilizado não apresenta resultados muito bons. Porém olhando para o 'log-likelihood ratio test' nos indica que modelos do mesmo tipo tem um ganho significativo se bem parametrizados.

# In[43]:


cph.print_summary(columns=["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "z", "p"], decimals = 4)


# Analisando as chances de sobrevivencias de dois individuos, percebemos que claramente o tempo é um fator decisivo na sua chance de sobrevivência.

# In[44]:


cph.predict_survival_function(heart.loc[1]).plot(title="Probabilidade de sobrevivencia do individuo 01 ao longo do tempo")


# In[45]:


cph.predict_survival_function(heart.loc[10]).plot(title="Probabilidade de sobrevivencia do individuo 10 ao longo do tempo")


# Abaixo é realizado uma analise cada variavel de forma individual, de forma que o nosso objetivo é ver o impacto de cara variavel desconsiderando as outras. Assim, o ideal é olhar para as variaveis de forma unificada como foi feito anteriormente olhando o resumo estatistico.

# É claramente perceptivel que aumentando a idade do individuo as suas chances de sobrevivencia são menores ao longo do tempo.
# 
# Um individuo de 35 anos após 250 dias, ainda teria quase 95% de chance de sobrevivencia, enquanto uma pessoa com 60 anos teria 76%, já uma pessoa com 75 anos teria 65%. Essa diferença vai aumentando cada vez mais apartir dos 60 anos, chegando até os 95 anos, onde no mesmo periodo de tempo uma pessoa teria 30% de chance de sobrevivencia.

# In[46]:


cph.plot_partial_effects_on_outcome(covariates = 'age', values = [35, 45, 55, 60, 65, 70, 75, 85, 95], cmap = 'coolwarm')


# É perceptivel que a maioria das pessoas com CPK no sangue até 1000 mcg/L possuem chances de sobrevivencia similares, porém com o seu aumento para 2000 já temos um grande pulo de menos 10% na chance de sobrevivencia, já para 4000 o nosso pulo é de mais de 20%. 
# 
# O que faz total sentido já que um exame normal deve constar valores entre 10 e 120 mcg/L no sangue. Valores alto constumam indicar riscos de ataqque cardiaco, convulsão, delirio...

# In[47]:


cph.plot_partial_effects_on_outcome(covariates = 'creatinine_phosphokinase', values = [250, 500, 1000, 2000, 3000, 4000],                                    cmap = 'coolwarm')


# Já a variavel 'ejection_fraction' é proporcinal as chances de sobrevivencia, ou seja seu aumento é positivo para o individuo, é esperado taxas entre 50% e 75%. Menos que 50% começa a apresentar riscos a saude, e abaixo de 40% apresenta graves riscos.

# In[48]:


cph.plot_partial_effects_on_outcome(covariates = 'ejection_fraction', values = [20, 30, 40, 50, 60, 70, 80], cmap = 'coolwarm')


# Contudo é perceptivel que a variavel 'platelets' não possui grande significancia tanto para o aujmento ou diminuição nas chances de sobrevivencia do individuo.

# In[49]:


cph.plot_partial_effects_on_outcome(covariates = 'platelets',                                    values = [200000, 300000, 400000, 500000, 600000, 700000, 800000],                                    cmap = 'coolwarm')


# A variavel 'serum_creatinine' aparenta possuir forte relação com as chances de sobrevivencia, visto que o seu valor em 1 mg/dl apresenta uma chance se sobrevivencia proximo a 90%, já em 3 mg/dl caimos para 75%, 5 mg/dl para 50% até chjegarmosa em 9 mg/dl com 10% de sobrevivencia.
# 
# Isso é totalmente normal, visto que o normal é um resultado entre 0.74 e 1.35 mg/dl para homens e 0.59 a 1.04 mg/dl para mulheres. Qualquer valor fora dessas margens já começa a ser preocupante, apresentando sinais de má função renal.

# In[50]:


cph.plot_partial_effects_on_outcome(covariates = 'serum_creatinine', values = [1, 2, 3, 4, 5, 6, 7, 8, 9], cmap = 'coolwarm')


# Por outro lado a variavel 'serum_sodium' aparenta possuir uma relação moderada com nosso alvo, considerando que valores maiores aumentam a taxa de sobrevivencia. Sendo esperado em um resultado normal valores entre 135 e 145 mEq/L. Sendo preocupante valores menores que 135 mEq/L, indicando riscos de fadiga, nausea, perda de consciência e até coma.

# In[51]:


cph.plot_partial_effects_on_outcome(covariates = 'serum_sodium', values = [125, 130, 135, 140, 145], cmap = 'coolwarm')


# In[52]:


def plot_partial_categorical(cph_, columns):
    for column in columns:
        cph_.plot_partial_effects_on_outcome(covariates = column, values = [0, 1], cmap = 'coolwarm')


# Visualizando a regressão para os dados categoricos é perceptivel que algumas variaveis não possuem tanto impacto em relação a outras.
# 
# A variavel 'Anaemia' possui ganho significativo. Pessoas sem 'Anaemia' possuem 10% a mais de chance de sobreviver.
# 
# A variavel 'Diabetes' não possui ganho significativo. Pessoas com 'Diabetes' não possuem chances muito maior de sobrevivencia.
# 
# A variavel 'high_blood_pressure' possui ganho significativo. Pessoas sem 'high_blood_pressure' possuem 10% a mais de chance de sobreviver.
# 
# A variavel 'sex' possui ganho pequeno. Pessoas do 'sex' 0 (Feminino) possuem  5% a menos de chance de sobreviver o 'sex' 1 (Masculino).
# 
# A variavel 'Smokinng' não possui ganho significativo. Pessoas que fumam não possuem chances muito maiores de sobrevivencia.
# 

# In[53]:


plot_partial_categorical(cph, categoricas)


# Devido ao time ser uma variavel que possui valores inexistentes em um cenario real, irá ser removida do dataset.

# In[54]:


continuas.remove('time')
heart = heart.drop('time', axis = 1)
heart.head()


# # 2. Avaliando a MultiColinearidade

# In[55]:


X = heart.iloc[:, :-1]
y = heart['DEATH_EVENT'].values
columns = heart.keys()


# ## 2.1 Autovetores

# In[56]:


X.head()


# Para avaliarmos a multicolinearidade iremos utilizar de correlação e atribuir aos metodos de autovetores. Onde iremos ter a variancia acumulada entre as variaveis. Para esse avaliação iremos utilizar somente as variaveis continuas.

# In[57]:


corr = np.corrcoef(X[continuas], rowvar = 0)
eigenvalues, eigenvectors = np.linalg.eig(corr)


# In[58]:


print(eigenvalues, min(eigenvalues))


# In[59]:


print(abs(eigenvectors[:, 2]))


# Captando as informações é dito que possui alta correlação entre as variaveis 'age', 'ejection_fraction', 'platelets', 'serum_creatinine' e 'serum_sodium'. Portanto é importante analisar se a multicolinearidade realmente existe, pois analisando o mapa de calor essa correlação não parece ser forte, chegando a no maximo 0.25.

# In[60]:


print(continuas[0], continuas[1], continuas[2], continuas[4], continuas[5])


# ## 2.2 Visualizando Multicolinearidade

# In[61]:


def scatter_plot_conjunto(data, columns, target):
    # Definindo range de Y
    y_range = [data[target].min(), data[target].max()]
    
    for column in columns:
        if target != column:
            # Definindo range de X
            x_range = [data[column].min(), data[column].max()]
            
            # Scatter plot de X e Y
            scatter_plot = data.plot(kind = 'scatter', x = column, y = target, xlim = x_range, ylim = y_range)
            
            # Traçar linha da media de X e Y
            meanX = scatter_plot.plot(x_range, [data[target].mean(), data[target].mean()], '--', color = 'red', linewidth = 1)
            meanY = scatter_plot.plot([data[column].mean(), data[column].mean()], y_range, '--', color = 'red', linewidth = 1)


# In[62]:


heart_multicolinearidade = heart[['age', 'creatinine_phosphokinase' ,'ejection_fraction', 'serum_creatinine', 'serum_sodium']]


# Analisando os scatterplots abaixos não é notavel uma grande correlação entre as variaveis indicadas, sendo assim não podendo determinar de imediato a existencia de multicolinearidade.

# In[63]:


sns.pairplot(heart_multicolinearidade)


# Algumas analises sobre cada variavel continua em relação com a variavel target.
# 
# Age: O seu valor tende a ser maior na classe 1, representado pela média e pelos quartis inferiores/superiores. Esse valor já era esperado devido ao aumento da vulnerabilidade na classe idosa.
# 
# Ejection_Fraction: O seu valor tende a ser menor na classe, representando um cenário onde pessoas que morrem por doenças cardiovasculares tem uma menor taxa de sangue saindo do corpo.
# 
# Serum_creatine: É visto que claramente o seu aumento no sangue leva a maiores taxas de falha cardiovascular. Esse é um fato que gera muita controversia apesar de já terem estudos que comprovem o fato, agora temos dados.
# 
# Serum_sodium: Conforme o seu valor desce maior a taxa de falha cardiovascular por doença cardiovascular. É perceptivel que apartir de valores menores que 135mEq/L a taxa da classe 1 aumenta muito, isso se da principalmente pela Hiponatremia.
# 
# Time: Claramente de acordo com o seu declineo possui um aumento avassalador na classe 1. O que faz total sentido visto que pessoas que visitam o médico mais tarde quando a doença já possui uma maior gravidade, tem menos tempo de reagir a um tratamento.

# In[64]:


def boxplot_plot_individual(data, columns, target):
    
    fig = plt.figure()
    fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
    fig.set_figheight(25)
    fig.set_figwidth(15)
    
    columns_adjust = ceil(len(columns))
    
    for i, column in enumerate(columns):
        if column != target:
            ax = fig.add_subplot(columns_adjust, 3, i + 1)
            sns.boxplot(x = target, y = column, data = data)
    
    plt.tight_layout()
    plt.show()


# In[65]:


boxplot_plot_individual(heart, continuas, 'DEATH_EVENT')


# Por ulitmo, é interessante analisar o scatterplot de todas variaveis continuas e não somente as que possuem multicolinearidade. Assim conseguimos ter um overview de nossas variaveis.

# In[66]:


sns.pairplot(heart[continuas])


# # 3. Pre-Processamento

# ## 3.1 Detectando Outliers

# Ao olharmos para a distribuição dos nossos dados para verificar se obedecem a uma distribuição normal Gaussiana, e tambem para entender a tendencia dos valores, positiva ou negativa, iremos utilizar a métrica de skew. Onde é possível verificar que os nosos dados em sua maioria obedecem a uma distribuição normal Gaussiana, alguns com tendencia positiva e outros negativos, com exceção da variavel 'serum_creatine' que aparenta não obedecer a uma distribuição normal.
# ![image.png](attachment:image.png)

# In[67]:


print(heart[continuas].skew())


# Para termos um contraste dos numeros apresentados, iremos trazer o histograma de cada variavel acima. É verificado que a variavel 'age' se aproxima de uma distribuição normal enquanto as outras variaveis em sua maioria seguem para uma simetria negativa.

# In[68]:


def hist_individual(data, columns):
    fig = plt.figure()
    fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    
    columns_adjust = ceil(len(columns) / 3)
    
    for i, column in enumerate(columns):
        ax = fig.add_subplot(columns_adjust, 3, i + 1)
        data[column].hist(label = column)
        plt.title(column)
        
    plt.tight_layout()  
    plt.show()


# In[69]:


hist_individual(heart, continuas)


# Uma das estatisticas que pode ser de grande importância para a detecção de outliers é Exceço de Kurtosis, onde podemos entender facilmente as variaveis que possuem muitos valores distuantes (outliers). Onde a sua formula é constituiada em:
# 
# #### Exceço de Kurtosis = Kurtosis - 3
# 
# Mesokurtic -> Exceço de Kurtosis ~= 0: Outliers proximo a distribuição normal.
# 
# Leptokurtic -> Exceço de Kurtosis < 0: Muitos outliers, onde estão tendo peso consideravel.
# 
# Platykurtic -> Exceço de Kurtosis > 0: Poucos outliers, onde não estão tendo um peso muito consideravel.

# In[70]:


print(heart[continuas].kurtosis() - 3)


# Para um melhor entendimento sobre os numeros oferecidos pela metrica de Exceço de Kurtosis, iremos trazer o boxplot de cada uma dessas variaveis. É perceptivel que nas variaveis 'age' e 'ejection_fraction', por possuirem uma kurtosis muito negativa os seus outliers são quase que imperceptiveis, visto que não trazem peso para o problema. Porém a variavel 'creatinine_phosphokinase', 'platelets' e 'serum_creatinine' possuem muitos outliers, trazendo grande peso ao boxplot.

# In[71]:


def boxplot_individuais(data, columns):
    fig = plt.figure()
    fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
    fig.set_figheight(8)
    fig.set_figwidth(15)
    
    columns_adjust = ceil(len(columns) / 3)
    
    for i, column in enumerate(columns):
        ax = fig.add_subplot(columns_adjust, 3, i + 1)
        sns.boxplot(x = data[column])
        
    plt.tight_layout()  
    plt.show()


# In[72]:


boxplot_individuais(heart, continuas)


# ## 3.2 Removendo Outliers

# In[73]:


# Criando array de colunas com outliers
outlier_columns = continuas.copy()
outlier_columns.remove('age')
print(outlier_columns)


# In[74]:


def outlier_log_transformation_based(data, columns):
    for column in columns:
        data[column] = data[column].map(lambda x: np.log(x) if x > 0 else 0)
    return data


# In[75]:


def outlier_percentil_based(data, columns):
    for column in columns:
        # Capturando percentile de 10 e 90
        percentil10 = data[column].quantile(0.10)
        percentil90 = data[column].quantile(0.90)
        
        data[column] = np.where(data[column] < percentil10, percentil10, data[column])
        data[column] = np.where(data[column] > percentil90, percentil90, data[column])
        
    return data


# In[76]:


heart = outlier_percentil_based(heart, outlier_columns)


# Analisando novamente o Skew e o histograma, é perceptivel que os dados perderam um pouco do seu formato.

# In[77]:


print(heart[continuas].skew())


# In[78]:


hist_individual(heart, continuas)


# Já analisando a Kurtosis é perceptivel que os dados tiveram os seus outliers ajustados, passando a não exibirem valores extremos com frequência.

# In[79]:


print(heart[continuas].kurtosis() - 3)


# In[80]:


boxplot_individuais(heart, continuas)


# ## 3.3 Padronização dos Dados

# Para os algoritmos que iremos aplicar como K-means e SVM o ideal é utilizar a Padronização ao invés da Normalização.

# In[81]:


X = heart.iloc[:, :-1]
y = heart['DEATH_EVENT'].values


# In[82]:


X.head()


# In[83]:


print(y)


# In[84]:


# Aplicando padronização
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)


# In[85]:


print(X_standard)


# In[86]:


plt.hist(X_standard[:,0:1])


# # 4.  Balanceando os Dados

# É demonstrado abaixo que os dados estão desbalanceados, com  aproximadamente dois terços da base sendo da classe 0. Assim o nosso modelo irá se tornar tendencioso, iremos tentar utilizar tecnicas de balanceamento como SMOTE para melhorar os resultados.

# In[87]:


heart_original.DEATH_EVENT.value_counts().plot(kind = 'bar', title = 'Count DEATH EVENT')


# In[88]:


x_train, x_test, y_train, y_test = train_test_split(X_standard, y, test_size = .3, random_state = seed_)


# In[89]:


oversample = SMOTE(random_state = seed_)
x_train_resample, y_train_resample = oversample.fit_resample(x_train, y_train)


# In[90]:


y_all = np.concatenate((y_train_resample, y_test), axis = 0)


# In[91]:


print(np.shape(y_test),
np.shape(y_train_resample),
np.shape(y_all))


# In[92]:


Counter(y_all)


# Após um balanceamento é notavel um aumento significativo de ocorrências na classe 1, que passa a conter 193 casos contra 203 para a classe 0.

# In[93]:


dt = pd.DataFrame(y_all, columns = ['target'])
dt.target.value_counts().plot(kind ='bar', title = 'Count DEATH EVENT')


# # 5. Feature Selecting

# Para aumentar a generalização do nosso modelo, evitando futuro overfitting, podemos aplicar de técnicas de Feature Selecting para a redução de algumas variaveis. Antes é importante ressaltar quais insights já foram dados ao longo da analise exploratória.
# 
# 1 - A variavel 'times' já foi removida do dataset e não será utilizada visto que possui ligação direta com a variavel target, sendo uma variavel coletada somente após o evento acontecer.
# 
# 2 - Age: O seu valor tende a ser maior na classe 1, representado pela média e pelos quartis inferiores/superiores. Esse valor já era esperado devido ao aumento da vulnerabilidade na classe idosa.
# 
# 3 - Ejection_Fraction: O seu valor tende a ser menor na classe, representando um cenário onde pessoas que morrem por doenças cardiovasculares tem uma menor taxa de sangue saindo do corpo.
# 
# 4 - Serum_creatine: É visto que claramente o seu aumento no sangue leva a maiores taxas de falha cardiovascular. Esse é um fato que gera muita controversia apesar de já terem estudos que comprovem o fato, agora temos dados.
# 
# 5 - Serum_sodium: Conforme o seu valor desce maior a taxa de falha cardiovascular por doença. É perceptivel que apartir de valores menores que 135mEq/L a taxa da classe 1 aumenta muito, isso se da principalmente pela Hiponatremia.
# 
# 6 - Aparenntemente as variaveis 'age', 'ejection_fraction', 'platelets', 'serum_creatinine' e 'serum_sodium' apresentam multicolinearidade, porém ao plotar as variaveis e visualizar os dados, não aparentam demonstrar muita correlação.

# ## 5.1 XGBOOST

# In[94]:


modeloXGB = xgb.XGBClassifier(n_estimators = 1000, use_label_encoder = False, seed_ = seed_)


# In[95]:


modeloXGB.fit(X_standard, y)


# In[96]:


print(modeloXGB.feature_importances_)


# In[97]:


index_ordenado = modeloXGB.feature_importances_.argsort()


# In[98]:


index_ordenado


# In[99]:


plt.barh(heart.drop('DEATH_EVENT', axis = 1).columns[index_ordenado], modeloXGB.feature_importances_[index_ordenado])


# In[100]:


plot_importance(modeloXGB)


# ## 5.2 RFE (Recursive Feature Elimination)

# In[101]:


modeloRFE = LogisticRegression(solver = 'lbfgs', random_state = seed_)
rfe = RFE(modeloRFE, n_features_to_select = 6)
fit = rfe.fit(X_standard, y)


# In[102]:


print("Features Selecionadas: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)


# In[103]:


selecionadas = [columns[i] for i, col in enumerate(fit.support_) if col == True]


# In[104]:


selecionadas


# In[105]:


heart[selecionadas].head()


# ## 5.3 Extra Trees Classifier

# In[106]:


modeloExtraTrees = ExtraTreesClassifier(n_estimators = 1000, random_state = seed_)
modeloExtraTrees.fit(X_standard, y)
print(modeloExtraTrees.feature_importances_)


# In[107]:


index_ordenado_extra = modeloExtraTrees.feature_importances_.argsort()


# In[108]:


plt.barh(heart.drop('DEATH_EVENT', axis = 1).columns[index_ordenado_extra],
         modeloExtraTrees.feature_importances_[index_ordenado_extra])


# ## 5.4 Aplicando Feature Selecting

# Analisando os 3 algoritmos de Feature Selection, sendo dois baseados em algoritmos ensemble e um em eliminação recursiva. Possuimos o seguinte cenário, os 3 modelos concordaram que as variaveis 'ejection_fraction', 'serum_creatine', 'serum_sodium' e 'age' devem ser incluidas. Após analisar a correlação das variaveis e suas distribuções foi optado por utilizar as 6 primeiras variaveis do modelo XGBOOST, além das 4 citadas anteriormente serão incluidas 'sex' e 'smoking' essas que apresentaram uma relevância importante para o modelo. Essas se encaixam nas analises também feitas estatisticamente, o que representa um bom sinal em relação a analise humana e a do algoritmo.

# In[109]:


heart.head()


# ### 5.4.1 Feature Selecting Baseado em Arvore - Mantendo Multicolineares

# In[110]:


'''
index_ordenado_invertido = np.flip(index_ordenado)
print(index_ordenado_invertido)

heart = heart.iloc[:, index_ordenado_invertido[0:6]]
features = index_ordenado_invertido[0:6]

# Mantendo os 6 melhores de acordo com XGBOOST sem excluir variaveis MultiColineares
x_train = x_train[:, index_ordenado_invertido[0:6]]
x_test = x_test[:, index_ordenado_invertido[0:6]]
x_train_resample = x_train_resample[:, index_ordenado_invertido[0:6]]
'''


# ### 5.4.2 Feature Selecting Baseado em Arvore - Excluindo Multicolineares

# In[111]:


'''
Mantendo os 6 melhores de acordo com XGBOOST excluindo as MultiColinearidades

Variaveis com multicolinearidade:
age, creatinine_phosphokinase, ejection_fraction, serum_creatinine e serum_sodium
'''
'''
features = [0, 2, 3, 5, 6, 8]

heart = heart.iloc[:, features]

x_train = x_train[:, features]
x_test = x_test[:, features]
x_train_resample = x_train_resample[:, features]
'''


# ### 5.4.3 Feature Selecting Baseado no Cox Proportional Hazard - Mantendo Multicolineares

# In[112]:


'''
Mantendo as variaveis indicadas confiaveis e ideais pelo Cox Proportional Hazard,
sem adaptar o modelo para remover multicolineres.
'''

features = [5, 1, 7, 3, 10, 0]

heart = heart.iloc[:, features]

x_train = x_train[:, features]
x_test = x_test[:, features]
x_train_resample = x_train_resample[:, features]


# ## 5.5 Visualizando Resultado do Feature Selecting

# In[113]:


heart.head()


# In[114]:


print(x_train[0])


# # 6. Modelagem Preditiva

# Iniciando o ciclo de modelagem preditiva o nosso objetivo é treinar dois modelos, o primeiro modelo será o K-Means e após o SVM, primeiramente iremos utilizar o K-means para entender se o nosso modelo consegue separar as classes da forma correta, dessa forma iremos ter um melhor overview para treinar um modelo mais complexo como SVM.

# In[115]:


def report_modelo(modelo, y, pred, label = 'Modelo', save = False, target_names = [0, 1], cut_limit = 0.5):
    # Forçando predições para um numero inteiro
    pred[pred > cut_limit] = 1
    pred[pred <= cut_limit] = 0
    
    # Plotando a matriz de confusão
    cm = confusion_matrix(y, pred)
    cm = pd.DataFrame(cm, index = target_names, columns= target_names)

    plt.figure(figsize = (10, 10))
    sns.heatmap(cm, cmap = "Blues", linecolor = 'black', linewidths = 1, annot = True,                 fmt = '', xticklabels = target_names, yticklabels = target_names)
    plt.show()
    
    print('AUC: %f' % roc_auc_score(y, pred))
    
    # Area sob  a curva ROC
    rfp, rvp, lim = roc_curve(y,  pred)

    plt.plot(rfp, rvp, marker = '.',  label = label,  color = 'orange')
    plt.plot([0, 1],  [0, 1], color = 'darkblue', linestyle = '--')
    plt.xlabel('Especificade')
    plt.ylabel('Sensibilidade')
    plt.legend()
    plt.show()
    
    # Acurácia
    print("Acurácia: %f" % accuracy_score(y, pred))
    
    # Classification Report
    print(classification_report(y, pred, target_names= target_names))    
    
    # Salvando modelo sem sobreescrever arquivos existentes
    if save:
        shortFileName = '000'
        fileName = 'models/0001.model'
        fileObj = Path(fileName)
        
        index = 1
        while fileObj.exists():
            index += 1
            fileName = 'models/' + shortFileName + str(index) + '.model'
            fileObj = Path(fileName)
        
        # salvar modelo
        pickle.dump(modelo, open(fileName, 'wb'))
        
        return fileName


# ## 6.1 K-Means

# Para o modelo K-means onde o K seria o numero de classe, não iremos realizar nenhum estudo sobre o numero de classes pois já sabemos com absoluta certeza quantas classe possuimos. Para esse problema, possuimos 2 classe, 0 ou 1, mais especificamente Morto ou Vivo. O unico objetivo com esse modelo é saber se os nossos dados possuem uma boa representatividade para algoritmos de Machine Learning.

# Uma métrica que iremos utilizar será 'Hopkins' que irá nos dizer se os nossos dados são clusterizaveis.
# 
# Valores > .5 significam que o dataset não é "clusterizável"
# 
# Valores < .5 significam que o dataset é "clusterizável"
# 
# Quanto mais próximo de zero melhor.

# In[116]:


# Hopkins sem padronização
print("Sem padronização:", hopkins(X, X.shape[0]))

# Hopkins com padronização
print("Com padronização:", hopkins(X_standard, X_standard.shape[0]))

# Hopkins com padronização e feature selecting
X_standard_feature = X_standard[:, features]
print("Com padronização e feature selecting:", hopkins(X_standard_feature, X_standard_feature.shape[0]))

# Hopkins com normalização
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("Com normalização:", hopkins(X_scaled, X_scaled.shape[0]))

# Hopkins com normalização e feature selecting
X_scaled_feature = X_scaled[:, features]
print("Com normalização e feature selecting:", hopkins(X_scaled_feature, X_scaled_feature.shape[0]))


# É perceptivel que os dados com normalização e feature selecting são os que possuem melhores resultado. Isso se da pois algoritmos como K-Means tendem a ter melhores resultados bons com normalização, baseando se na distância. Assim a normalização ficou com 0.14 e a padronização 0.22.

# In[117]:


print(x_train.shape, x_test.shape)


# In[118]:


x_full = np.concatenate([x_train, x_test])


# In[119]:


print(x_full.shape)


# In[120]:


# Aplicando a redução de dimensionalidade
pca = PCA(n_components = 2)
pca = pca.fit_transform(x_full)


# Para o primeiro modelo iremos utilizar as variaveis de x sem o balanceamento nos dados, pois esse fator não deve ter um grande impacto, visto que precisamos separar os clusters e não os classificar.

# In[121]:


modelo_v1 = KMeans(n_clusters = 2, random_state = seed_)
modelo_v1.fit(pca)


# In[122]:


x_min, x_max, y_min, y_max, xx, yy, Z = [0, 0, 0, 0, 0, 0, 0]


# In[123]:


def minMax(pca_, modelo):
    global x_min, x_max, y_min, y_max, xx, yy, Z
    
    # Obtenção de valores minimos e maximos
    x_min, x_max = pca_[:, 0].min(), pca_[:, 0].max()
    y_min, y_max = pca_[:, 1].min(), pca_[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)


# In[124]:


minMax(pca, modelo_v1)


# In[125]:


def areaCluster():
    # Plot das areas dos clusters
    plt.imshow(Z, interpolation = 'nearest',
               extent = (xx.min(), xx.max(), yy.min(), yy.max()),
               cmap = plt.cm.Paired,
               aspect = 'auto',
               origin = 'lower')


# In[126]:


areaCluster()


# In[127]:


def plotCentroides(pca_, modelo):
    # Plot dos centroides
    plt.plot(pca_[:, 0], pca_[:, 1], 'k.', markersize = 4)
    centroids = modelo.cluster_centers_
    inert = modelo.inertia_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 169, linewidths = 3, color = 'r', zorder = 8)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


# In[128]:


plotCentroides(pca, modelo_v1)


# O Silhouette score varia entre -1 e 1, quanto mais próximo de 1 melhor.

# In[129]:


get_ipython().run_line_magic('pinfo', 'silhouette_score')


# In[130]:


# Silhouette Score
labels = modelo_v1.labels_
silhouette_score(pca, labels, metric = 'euclidean')


# O segundo modelo iremos utilizar dados balanceados.

# In[131]:


x_full_resample = np.concatenate([x_train_resample, x_test])


# In[132]:


# Aplicando a redução de dimensionalidade
pca2 = PCA(n_components = 2)
pca2 = pca2.fit_transform(x_full_resample)


# In[133]:


modelo_v2 = KMeans(n_clusters = 2, random_state = seed_)
modelo_v2.fit(pca2)


# In[134]:


minMax(pca2, modelo_v2)


# In[135]:


areaCluster()


# In[136]:


plotCentroides(pca2, modelo_v2)


# Para a separação dos dados entre dois clusters, foi visto que o nosso modelo teve uma piora significativa em relação ao anterior após utilizar dados gerados artificialmente para forçar o balancemanto.

# In[137]:


# Silhouette Score
labels = modelo_v2.labels_
silhouette_score(pca2, labels, metric = 'euclidean')


# In[138]:


# Modelo v3
# Iremos utilizados os dados normalizados com feature selecting

# Aplicando a redução de dimensionalidade
pca3 = PCA(n_components = 2)
pca3 = pca3.fit_transform(X_scaled_feature)

modelo_v3 = KMeans(n_clusters = 2, random_state = seed_)
modelo_v3.fit(pca3)

minMax(pca3, modelo_v3)


# In[139]:


areaCluster()


# In[140]:


plotCentroides(pca3, modelo_v3)


# In[141]:


# Silhouette Score
labels = modelo_v3.labels_
silhouette_score(pca3, labels, metric = 'euclidean')


# Analisando os resultados acima é tomada algumas conclusões. Primeiramente, como esperado não pode se esperar que um conjunto de dados para classificação possua bons resultados para clusterização. Temos isso representado no Silhouette Score de 0.41 do terceiro modelo, representando que o cluster possui tendência a não ser clusterizavel. Porém iremos seguir para um algoritmo mais complexo tentando extrair as previsões dos dados que possuimos.
# 
# Já é alertado que um dos problemas que iremos ter posteriormente é a pouca quantidade de dados, o que irá dificultar a aprendizagem do algoritmo.

# ## Observação
# 
# Irá ser adotado a nomeclatura 'VIVER' para o paciente que não tiver uma falha cardiovascular e 'MORRER' para o que tiver, essa nomeclatura foi adotada pois lendo artigos e outras submissões de notebooks do mesmo tipo, é preferivel adotar essa nomeclatura, já que uma falha cardiovascular desse tipo teria altas chances de levar o paciente a falecer.
# 
# Essa nomeclatura também tem relacão com o nome da variavel no dataset, 'DEATH_EVENT'.

# ## 6.2 SVM

# Abaixo iniciamos a modelagem utilizando o algoritmo SVM, posteriormente ainda iremos testar com o algoritmo XGBoost. Entretanto o notebook não terá mais tantas anotações, isso se da pois é uma etapa de modelagem e treinamento intenso. Caso prefira, no final do notebook ainda tem o item 7 abordando as conclusões e apresentando os resultados do modelo final utilizado.

# In[142]:


# Criação do modelo v1
modelo_svm_v1 = SVC(kernel = 'linear', random_state = seed_)


# In[143]:


# Criando base sem padronização, balanceamento e feature selecting
x_train_nothing, x_test_nothing, y_train_nothing, y_test_nothing =                                                train_test_split(X, y, test_size = .3, random_state = seed_)


# In[144]:


# Treinamento
start = time.time()
modelo_svm_v1.fit(x_train_nothing, y_train_nothing)
end = time.time()
print('Tempo de Treinamento do Modelo:', round(end - start, 4))


# In[145]:


pred_v1 = modelo_svm_v1.predict(x_test_nothing)


# In[146]:


report_modelo(modelo_svm_v1, y_test_nothing, pred_v1, label = 'SVM V1', target_names = ['VIVER', 'MORRER'])


# In[147]:


# Criação do modelo v2
modelo_svm_v2 = SVC(kernel = 'linear', random_state = seed_)


# In[148]:


# Criando base sem balanceamento e feature selecting. Com padronização
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train_nothing)
x_test_sc = sc.fit_transform(x_test_nothing)


# In[149]:


# Treinamento
start = time.time()
modelo_svm_v2.fit(x_train_sc, y_train_nothing)
end = time.time()
print('Tempo de Treinamento do Modelo:', round(end - start, 4))


# In[150]:


pred_v2 = modelo_svm_v2.predict(x_test_sc)


# In[151]:


report_modelo(modelo_svm_v2, y_test_nothing, pred_v2, label = 'SVM V2', target_names = ['VIVER', 'MORRER'])


# In[152]:


# Criação do modelo v3
modelo_svm_v3 = SVC(kernel = 'linear', random_state = seed_)


# In[153]:


# Criando sem feature selecting. Com padronização e balanceamento
oversample2 = SMOTE(random_state = seed_)
x_train_sc_resample, y_train_resample_2 = oversample2.fit_resample(x_train_sc, y_train_nothing)


# In[154]:


# Treinamento
start = time.time()
modelo_svm_v3.fit(x_train_sc_resample, y_train_resample_2)
end = time.time()
print('Tempo de Treinamento do Modelo:', round(end - start, 4))


# In[155]:


pred_v3 = modelo_svm_v3.predict(x_test_sc)


# In[156]:


report_modelo(modelo_svm_v3, y_test_nothing, pred_v3, label = 'SVM V3', target_names = ['VIVER', 'MORRER'])


# In[157]:


# Criação do modelo v4
modelo_svm_v4 = SVC(kernel = 'linear', random_state = seed_)


# In[158]:


# Criando com padronização, balanceamento e feature selecting
# x_train_resample e x_test já possuem essas caracteristicas


# In[159]:


# Treinamento
start = time.time()
modelo_svm_v4.fit(x_train_resample, y_train_resample)
end = time.time()
print('Tempo de Treinamento do Modelo:', round(end - start, 4))


# In[160]:


pred_v4 = modelo_svm_v4.predict(x_test)


# In[161]:


report_modelo(modelo_svm_v4, y_test, pred_v4, label = 'SVM V4', target_names = ['VIVER', 'MORRER'])


# É visualizado que os modelos que mais se destacam são a Versão 01 e 04. Em que que a versão 01 esta muito orientada pelos individios que irão morrer. Contudo, o nosso foco é saber quem irá morrer nas condições atuais, assim podemos evitar a falha cardiovascular. Para isso o modelo da Versão 04 teve uma melhora significativa.
# 
# Agora iremos utilizar o grid e random search cv, para termos uma ideia inicial dos melhores hiperparametros do algoritmo.

# In[162]:


def treina_GridSearchCV(modelo, params_, x_treino, y_treino, x_teste, y_teste,                        n_jobs = 20, cv = 5, refit = True, title = 'SVM', scoring = None, salvar_resultados = False,                       report_treino = False):
    grid = GridSearchCV(modelo, params_, n_jobs = n_jobs, cv = cv, refit = refit, scoring = scoring)
    
    grid.fit(x_treino, y_treino)
    pred = grid.predict(x_teste)
    modelo_ = grid.best_estimator_
    
    print(grid.best_params_)
    
    target_names = ['VIVER', 'MORRER']
    
    print('Report Para Dados de Teste')
    
    report_modelo(modelo_, y_teste, pred, label = title, target_names = target_names)
    
    if report_treino:
        print('Report Para Dados de Treino')
        pred_treino = grid.predict(x_treino)
        
        # Acurácia
        print("Acurácia: %f" % accuracy_score(y_treino, pred_treino))
          
        # Classification Report
        print(classification_report(y_treino, pred_treino, target_names= target_names))    
    
    if salvar_resultados:
        resultados_df = pd.DataFrame(grid.cv_results_)
        
        return resultados_df 


# In[163]:


# Criação do modelo intenso 05

params = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.9, 1.0, 1.1],
    'gamma': ['scale', 'auto'],
    'class_weight': ['balanced'],
    'random_state': [seed_]
}


# In[164]:


get_ipython().run_cell_magic('time', '', "treina_GridSearchCV(SVC(), params, x_train_resample, y_train_resample, x_test, y_test, title = 'SVM V5', cv = 10)")


# In[165]:


# Criação do modelo intenso 06

params = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.9, 1.0, 1.1],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'class_weight': ['balanced', {0:1, 1:5}, {0:1, 1:10}],
    'random_state': [seed_]
}


# In[166]:


get_ipython().run_cell_magic('time', '', "treina_GridSearchCV(SVC(), params, x_train_resample, y_train_resample, x_test, y_test, title = 'SVM V6')")


# In[167]:


get_ipython().run_cell_magic('time', '', "\n# Criação do modelo 07\nmodelo_svm_v7 = SVC(C = 1.1, class_weight = 'balanced', gamma = 'scale', kernel = 'rbf', random_state = seed_)\n\n# Treinamento\nmodelo_svm_v7.fit(x_train_resample, y_train_resample)\n\n# Previsão\npred_v7 = modelo_svm_v7.predict(x_test)\n\n# Report Geral\nreport_modelo(modelo_svm_v7, y_test, pred_v7, label = 'SVM V7', target_names = ['VIVER', 'MORRER'])")


# In[168]:


get_ipython().run_cell_magic('time', '', "\n# Criação do modelo 08\nmodelo_svm_v8 = SVC(C = 1000, class_weight = {0:1, 1:10}, gamma = 0.01, kernel = 'rbf', random_state = seed_)\n\n# Treinamento\nmodelo_svm_v8.fit(x_train_resample, y_train_resample)\n\n# Previsão\npred_v8 = modelo_svm_v8.predict(x_test)\n\n# Report Geral\nreport_modelo(modelo_svm_v8, y_test, pred_v8, label = 'SVM V8', target_names = ['VIVER', 'MORRER'])")


# In[169]:


class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_resample), y_train_resample)


# In[170]:


class_weights


# In[171]:


get_ipython().run_cell_magic('time', '', "\n# Criação do modelo 09\nmodelo_svm_v9 = SVC(C = 1000, class_weight = {0: 1, 1: 1}, gamma = 0.01, kernel = 'rbf', random_state = seed_)\n\n# Treinamento\nmodelo_svm_v9.fit(x_train_resample, y_train_resample)\n\n# Previsão\npred_v9 = modelo_svm_v9.predict(x_test)\n\n# Report Geral\nreport_modelo(modelo_svm_v9, y_test, pred_v9, label = 'SVM V9', target_names = ['VIVER', 'MORRER'])")


# In[172]:


# Criação do modelo intenso 10

params = {
    'kernel': ['rbf'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 1, 10, 100],
    'class_weight': ['balanced'],
    'random_state': [seed_]
}


# In[173]:


get_ipython().run_cell_magic('time', '', "treina_GridSearchCV(SVC(), params, x_train_resample, y_train_resample, x_test, y_test,\\\n                    title = 'SVM V10', scoring = 'top_k_accuracy')")


# In[174]:


# Criação do modelo intenso 11

params = {
    'kernel': ['rbf'],
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'gamma': ['scale', 'auto', 1, 10, 100],
    'class_weight': ['balanced'],
    'random_state': [seed_]
}


# In[175]:


get_ipython().run_cell_magic('time', '', "resultados = treina_GridSearchCV(SVC(), params, x_train_resample, y_train_resample, x_test, y_test,\\\n                    title = 'SVM V11', scoring = 'top_k_accuracy', salvar_resultados = True)")


# In[176]:


resultados[['param_C', 'param_class_weight', 'param_gamma', 'param_kernel',            'mean_test_score', 'std_test_score', 'rank_test_score']]


# In[177]:


get_ipython().run_cell_magic('time', '', '\n# Criação do modelo 12\nmodelo_svm_v12 = SVC(C = 100, class_weight = {0: 1, 1: 1.3}, gamma = 0.0001, kernel = \'rbf\', random_state = seed_)\n\n# Treinamento\nmodelo_svm_v12.fit(x_train_resample, y_train_resample)\n\n# Previsão\npred_v12 = modelo_svm_v12.predict(x_test)\n\n# Report Geral\nreport_modelo(modelo_svm_v12, y_test, pred_v12, label = \'SVM V12\', target_names = [\'VIVER\', \'MORRER\'])\n\nprint(\'Report Para Dados de Treino\\n\')\n\npred_treino = modelo_svm_v12.predict(x_train_resample)\n\nprint("Acurácia: %f" % accuracy_score(y_train_resample, pred_treino))\n\n# Classification Report\nprint(classification_report(y_train_resample, pred_treino, target_names= [\'VIVER\', \'MORRER\'])) ')


# In[178]:


# Criação do modelo intenso 12

params = {
    'kernel': ['rbf'],
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'gamma': [0.001, 0.01, 1, 10, 100],
    'class_weight': ['balanced'],
    'random_state': [seed_]
}


# In[179]:


get_ipython().run_cell_magic('time', '', "resultados = treina_GridSearchCV(SVC(), params, x_train_resample, y_train_resample, x_test, y_test,\\\n                    title = 'SVM V12', scoring = 'top_k_accuracy', salvar_resultados = True, report_treino = True)")


# In[180]:


resultados[['param_C', 'param_class_weight', 'param_gamma', 'param_kernel',            'mean_test_score', 'std_test_score', 'rank_test_score']]


# Após inumeros testes é perceptivel que os melhores hiperparametros escolhidos pelo GridSearchCV são:
# 
# {'C': 0.0001, 'class_weight': 'balanced', 'gamma': 'scale', 'kernel': 'rbf', 'random_state': 194}
# 
# O nosso modelo tende que a aumentar o recall para a classe 1 (MORRER) quando aumenta o 'C' e diminui o 'Gamma', porém o Recall também cai para a classe 0 (Viver). Tentando ajustar os parâmetros para ter um tradeoff ideal, não se mostrou favoravel. O problema esta na quantidade de dados.

# ## 6.3 XGBOOST

# O modelo de SVM não conseguiu encontrar os melhores vetores de suporte para separar os nossos dados com uma margem consideravel. Analisando os resultados o nosso modelo não aparenta estar sofrendo de overfitting. O nosso problema esta no tamanho do nosso dataset, possuimos poucos dados, para tentar melhorar o nosso Recall, iremos utilizar o algoritmo XGBoost, esse possui desempenhos melhores em dataset menores, pois combina multiplos algoritmos para realizar as predições, esse que pertence a familia dos ensemble.

# In[181]:


# Transformando os dados em DMatrix pois o XGBoost exige
dtrain_nothing = xgb.DMatrix(x_train_nothing, label = y_train_nothing)
dtest_nothing = xgb.DMatrix(x_test_nothing, label = y_test_nothing)


# In[182]:


# Definindo parametros e configurações
param = {}


# In[183]:


# Criação do modelo base v1
# Criando base sem padronização, balanceamento e feature selecting
modelo_xgb_v1 = xgb.train(params = param, dtrain = dtrain_nothing)

pred_v1 = modelo_xgb_v1.predict(dtest_nothing)

report_modelo(modelo_xgb_v1, y_test_nothing, pred_v1, label = 'XGB V1', target_names = ['VIVER', 'MORRER'])


# In[184]:


# Transformando os dados em DMatrix pois o XGBoost exige
dtrain_sc = xgb.DMatrix(x_train_sc, label = y_train_nothing)
dtest_sc = xgb.DMatrix(x_test_sc, label = y_test_nothing)

# Criação do modelo base v2
# Criando base sem balanceamento e feature selecting. Com padronização
modelo_xgb_v2 = xgb.train(params = param, dtrain = dtrain_sc)

pred_v2 = modelo_xgb_v2.predict(dtest_sc)

report_modelo(modelo_xgb_v2, y_test_nothing, pred_v2, label = 'XGB V2', target_names = ['VIVER', 'MORRER'])


# In[185]:


# Transformando os dados em DMatrix pois o XGBoost exige
dtrain_sc_resample = xgb.DMatrix(x_train_sc_resample, label = y_train_resample_2)
dtest_sc = xgb.DMatrix(x_test_sc, label = y_test_nothing)

# Criação do modelo base v3
# Criando sem feature selecting. Com padronização e balanceamento
modelo_xgb_v3 = xgb.train(params = param, dtrain = dtrain_sc_resample)

pred_v3 = modelo_xgb_v3.predict(dtest_sc)

report_modelo(modelo_xgb_v3, y_test_nothing, pred_v3, label = 'XGB V3', target_names = ['VIVER', 'MORRER'])


# In[186]:


# Transformando os dados em DMatrix pois o XGBoost exige
dtrain = xgb.DMatrix(x_train_resample, label = y_train_resample)
dtest = xgb.DMatrix(x_test, label = y_test)

# Criação do modelo base v4
# Criando com padronização, balanceamento e feature selecting
modelo_xgb_v4 = xgb.train(params = param, dtrain = dtrain)

pred_v4 = modelo_xgb_v4.predict(dtest)

report_modelo(modelo_xgb_v4, y_test_nothing, pred_v4, label = 'XGB V4', target_names = ['VIVER', 'MORRER'])


# Para o algoritmo XGBoost o modelo que melhor apresentou resultado foi a terceira versão sem feature selecting,com padronização e balanceamento. Esse modelo também apresentou um aumento na sua Especificidade podendo causar um futuro overfitting. Iremos tentar otimizar os parâmetros evitando o overfitting.

# In[187]:


params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.3, 0.2, 0.1, 0.05],
        'nthread': [2]
        }


# In[188]:


get_ipython().run_cell_magic('time', '', "# Criação do modelo v5\n# Criando sem feature selecting. Com padronização e balanceamento\n\ntreina_GridSearchCV(xgb.XGBClassifier(use_label_encoder = False), params, x_train_sc_resample, y_train_resample_2,\\\n                                 x_test_sc, y_test_nothing, title = 'XGB V05', report_treino = True)")


# In[189]:


params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.3, 0.2, 0.1, 0.05],
        'nthread': [2]
        }


# In[190]:


get_ipython().run_cell_magic('time', '', "# Criação do modelo v6\n# Criando com padronização, balanceamento e feature selecting\n\ntreina_GridSearchCV(xgb.XGBClassifier(use_label_encoder = False), params, x_train_resample, y_train_resample,\\\n                                 x_test, y_test, title = 'XGB V06', report_treino = True)")


# In[191]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n        'min_child_weight': [1, 2, 3],\n        'gamma': [1, 1.5, 2, 3],\n        'subsample': [0.6, 0.8, 1.0],\n        'colsample_bytree': [0.6, 0.8, 1.0],\n        'max_depth': [2, 3, 4, 5],\n        'learning_rate': [0.4, 0.3, 0.2, 0.1, 0.05],\n        'nthread': [2]\n        }\n\n# Criação do modelo v7\n# Criando com padronização, balanceamento e feature selecting\ntreina_GridSearchCV(xgb.XGBClassifier(use_label_encoder = False), params, x_train_resample, y_train_resample,\\\n                                 x_test, y_test, title = 'XGB V07', report_treino = True)")


# In[192]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n        'min_child_weight': [1, 2, 3, 4, 5, 6],\n        'gamma': [0.5, 0.7, 1, 1.5, 2, 3],\n        'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],\n        'colsample_bytree': [0.6, 0.8, 1.0, 1.2],\n        'max_depth': [5, 6, 7, 8, 9],\n        'learning_rate': [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],\n        'nthread': [2]\n        }\n\n# Criação do modelo v8\n# Criando com padronização, balanceamento e feature selecting\nmodelo_grid = xgb.XGBClassifier(use_label_encoder = False, early_stopping_rounds = 10, num_boost_round = 999)\ntreina_GridSearchCV(modelo_grid, params, x_train_resample, y_train_resample,\\\n                                 x_test, y_test, title = 'XGB V08', report_treino = True)")


# In[193]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n        'min_child_weight': [2, 3, 4, 5],\n        'gamma': [0.7, 1.3, 1.5, 1.7],\n        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],\n        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],    \n        'max_depth': [5, 6, 7],\n        'learning_rate': [0.5, 0.1, 0.01],\n        'nthread': [2],\n        'n_estimators' : [500]\n        }\n\n# Criação do modelo v9\n# Criando com padronização, balanceamento e feature selecting\nmodelo_grid = xgb.XGBClassifier(use_label_encoder = False)\n\nresultados = treina_GridSearchCV(modelo_grid, params, x_train_resample, y_train_resample,\\\n                                 x_test, y_test, title = 'XGB V09', report_treino = True,\\\n                                 salvar_resultados = True)")


# In[194]:


resultados


# In[195]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n        'min_child_weight': [2, 3, 4, 5],\n        'gamma': [0.7, 1.3, 1.4, 1.5],\n        'subsample': [0.6, 0.8, 0.9, 1.0],\n        'colsample_bytree': [0.8, 0.9, 1.0],    \n        'max_depth': [5, 6, 7],\n        'learning_rate': [0.5, 0.1, 0.01],\n        'lambda': [0.9, 1, 1.1],\n        'alpha': [0, 0.1],\n        'nthread': [2],\n        'n_estimators' : [500]\n        }\n\n# Criação do modelo v10\n# Criando com padronização, balanceamento e feature selecting\nmodelo_grid = xgb.XGBClassifier(use_label_encoder = False)\n\nresultados = treina_GridSearchCV(modelo_grid, params, x_train_resample, y_train_resample,\\\n                                 x_test, y_test, title = 'XGB V10', report_treino = True,\\\n                                 salvar_resultados = True)")


# In[196]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n        'min_child_weight': [1, 2, 3, 4, 5],\n        'gamma': [0, 0.1, 0.2, 0.3],\n        'subsample': [0.6, 0.8, 0.9, 1.0],\n        'colsample_bytree': [0.8, 0.9, 1.0],    \n        'max_depth': [5, 6, 7],\n        'learning_rate': [0.5, 0.1, 0.01],\n        'nthread': [2],\n        'n_estimators' : [500]\n        }\n\n# Criação do modelo v11\n# Criando com padronização, balanceamento e feature selecting\nmodelo_grid = xgb.XGBClassifier(use_label_encoder = False)\n\nresultados = treina_GridSearchCV(modelo_grid, params, x_train_resample, y_train_resample,\\\n                                 x_test, y_test, title = 'XGB V11', report_treino = True,\\\n                                 salvar_resultados = True)")


# In[197]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n        'min_child_weight': [1, 2, 3, 4],\n        'gamma': [0.7, 1.3, 1.4, 1.5],\n        'subsample': [0.6, 0.8, 0.9, 1.0],\n        'colsample_bytree': [0.8, 0.9, 1.0],\n        'colsample_bylevel': [0.8, 0.9, 1.0],\n        'max_depth': [5, 6, 7],\n        'learning_rate': [0.5, 0.1, 0.01],\n        'nthread': [2],\n        'n_estimators' : [500]\n        }\n\n# Criação do modelo v12\n# Criando com padronização, balanceamento e feature selecting\nmodelo_grid = xgb.XGBClassifier(use_label_encoder = False)\n\nresultados = treina_GridSearchCV(modelo_grid, params, x_train_resample, y_train_resample,\\\n                                 x_test, y_test, title = 'XGB V12', report_treino = True,\\\n                                 salvar_resultados = True)")


# In[198]:


get_ipython().run_cell_magic('time', '', "\nparams = {\n        'min_child_weight': [1, 2, 3, 4],\n        'gamma': [0.7, 1.3, 1.4, 1.5],\n        'subsample': [0.6, 0.8, 0.9, 1.0],\n        'colsample_bytree': [0.8, 0.9, 1.0],\n        'colsample_bynode': [0.8, 0.9, 1.0],\n        'max_depth': [5, 6, 7],\n        'learning_rate': [0.5, 0.1, 0.01],\n        'nthread': [2],\n        'n_estimators' : [500]\n        }\n\n# Criação do modelo v13\n# Criando com padronização, balanceamento e feature selecting\nmodelo_grid = xgb.XGBClassifier(use_label_encoder = False)\n\nresultados = treina_GridSearchCV(modelo_grid, params, x_train_resample, y_train_resample,\\\n                                 x_test, y_test, title = 'XGB V13', report_treino = True,\\\n                                 salvar_resultados = True)")


# In[199]:


#Transformando os dados em DMatrix pois o XGBoost exige
dtrain = xgb.DMatrix(x_train_resample, label = y_train_resample)
dtest = xgb.DMatrix(x_test, label = y_test)

params = {'colsample_bynode': 0.3, 'colsample_bytree': 1.0, 'gamma': 1.5, 'learning_rate': 0.5, 'max_depth': 10,          'min_child_weight': 1, 'n_estimators': 1000, 'nthread': 2, 'subsample': 0.8}

# Criação do modelo base v14
# Criando com padronização, balanceamento e feature selecting
modelo_xgb_v14 = xgb.train(params = params, dtrain = dtrain)

pred_v14 = modelo_xgb_v14.predict(dtest)

report_modelo(modelo_xgb_v14, y_test_nothing, pred_v14, label = 'XGB V14', target_names = ['VIVER', 'MORRER'])


# In[200]:


#Transformando os dados em DMatrix pois o XGBoost exige
dtrain = xgb.DMatrix(x_train_resample, label = y_train_resample)
dtest = xgb.DMatrix(x_test, label = y_test)

params = {'colsample_bynode': 0.3, 'colsample_bytree': 1.0, 'gamma': 2.3, 'learning_rate': 0.5, 'max_depth': 12,          'min_child_weight': 1, 'n_estimators': 500, 'nthread': 2, 'subsample': 0.8}

# Criação do modelo base v15
# Criando com padronização, balanceamento e feature selecting
modelo_xgb_v15 = xgb.train(params = params, dtrain = dtrain)

pred_v15 = modelo_xgb_v15.predict(dtest)

report_modelo(modelo_xgb_v15, y_test_nothing, pred_v15, label = 'XGB V15', target_names = ['VIVER', 'MORRER'])


# In[201]:


get_ipython().run_cell_magic('time', '', '\n#Transformando os dados em DMatrix pois o XGBoost exige\ndtrain_sc_resample = xgb.DMatrix(x_train_sc_resample, label = y_train_resample_2)\ndtest_sc = xgb.DMatrix(x_test_sc, label = y_test_nothing)\n\nparams = {\'colsample_bynode\': 0.6, \'colsample_bytree\': 0.3, \'gamma\': 2.3, \'learning_rate\': 0.5, \'max_depth\': 12,\\\n          \'min_child_weight\': 1, \'n_estimators\': 500, \'nthread\': 2, \'subsample\': 0.8}\n\n# Criação do modelo base v16\n# Criando sem feature selecting. Com padronização e balanceamento\nmodelo_xgb_v16 = xgb.train(params = params, dtrain = dtrain_sc_resample)\n\npred_v16 = modelo_xgb_v16.predict(dtest_sc)\n\nreport_modelo(modelo_xgb_v16, y_test_nothing, pred_v16, label = \'XGB V16\', target_names = [\'VIVER\', \'MORRER\'])\n\nprint(\'Report Para Dados de Treino\')\npred_treino = modelo_xgb_v16.predict(dtrain_sc_resample)\npred_treino = pred_treino > 0.5\n\n# Acurácia\nprint("Acurácia: %f" % accuracy_score(y_train_resample_2, pred_treino))\n\n# Classification Report\nprint(classification_report(y_train_resample_2, pred_treino, target_names= [\'VIVER\', \'MORRER\']))  ')


# In[202]:


get_ipython().run_cell_magic('time', '', '\n#Transformando os dados em DMatrix pois o XGBoost exige\ndtrain_sc_resample = xgb.DMatrix(x_train_sc_resample, label = y_train_resample_2)\ndtest_sc = xgb.DMatrix(x_test_sc, label = y_test_nothing)\n\nparams = {\'colsample_bynode\': 0.6, \'colsample_bytree\': 0.5, \'gamma\': 2.3, \'learning_rate\': 0.5, \'max_depth\': 12,\\\n          \'min_child_weight\': 1, \'n_estimators\': 500, \'nthread\': 2, \'subsample\': 0.8}\n\n# Criação do modelo base v17\n# Criando sem feature selecting. Com padronização e balanceamento\nmodelo_xgb_v17 = xgb.train(params = params, dtrain = dtrain_sc_resample)\n\npred_v17 = modelo_xgb_v17.predict(dtest_sc)\n\nreport_modelo(modelo_xgb_v17, y_test_nothing, pred_v17, label = \'XGB V17\', target_names = [\'VIVER\', \'MORRER\'])\n\nprint(\'Report Para Dados de Treino\')\npred_treino = modelo_xgb_v17.predict(dtrain_sc_resample)\npred_treino = pred_treino > 0.5\n\n# Acurácia\nprint("Acurácia: %f" % accuracy_score(y_train_resample_2, pred_treino))\n\n# Classification Report\nprint(classification_report(y_train_resample_2, pred_treino, target_names= [\'VIVER\', \'MORRER\']))  ')


# In[203]:


get_ipython().run_cell_magic('time', '', '\n#Transformando os dados em DMatrix pois o XGBoost exige\ndtrain_sc_resample = xgb.DMatrix(x_train_sc_resample, label = y_train_resample_2)\ndtest_sc = xgb.DMatrix(x_test_sc, label = y_test_nothing)\n\nparams = {\'colsample_bynode\': 0.6, \'colsample_bytree\': 0.5, \'gamma\': 2.3, \'learning_rate\': 0.5, \'max_depth\': 12,\\\n          \'min_child_weight\': 1, \'nthread\': 2, \'subsample\': 0.789}\n\n# Criação do modelo base v18\n# Criando sem feature selecting. Com padronização e balanceamento\nmodelo_xgb_v18 = xgb.train(params = params, dtrain = dtrain_sc_resample)\n\npred_v18 = modelo_xgb_v18.predict(dtest_sc)\n\nreport_modelo(modelo_xgb_v18, y_test_nothing, pred_v18, label = \'XGB V18\', target_names = [\'VIVER\', \'MORRER\'])\n\nprint(\'Report Para Dados de Treino\')\npred_treino = modelo_xgb_v18.predict(dtrain_sc_resample)\npred_treino = pred_treino > 0.5\n\n# Acurácia\nprint("Acurácia: %f" % accuracy_score(y_train_resample_2, pred_treino))\n\n# Classification Report\nprint(classification_report(y_train_resample_2, pred_treino, target_names= [\'VIVER\', \'MORRER\']))  ')


# In[204]:


get_ipython().run_cell_magic('time', '', '\n#Transformando os dados em DMatrix pois o XGBoost exige\ndtrain_sc_resample = xgb.DMatrix(x_train_sc_resample, label = y_train_resample_2)\ndtest_sc = xgb.DMatrix(x_test_sc, label = y_test_nothing)\n\nparams = {\'colsample_bynode\': 0.6, \'colsample_bytree\': 0.5, \'gamma\': 2.2, \'learning_rate\': 0.5, \'max_depth\': 12,\\\n          \'min_child_weight\': 1, \'nthread\': 2, \'subsample\': 0.789}\n\n# Criação do modelo base v19\n# Criando sem feature selecting. Com padronização e balanceamento\nmodelo_xgb_v19 = xgb.train(params = params, dtrain = dtrain_sc_resample)\n\npred_v19 = modelo_xgb_v19.predict(dtest_sc)\n\nreport_modelo(modelo_xgb_v19, y_test_nothing, pred_v19, label = \'XGB V19\', target_names = [\'VIVER\', \'MORRER\'])\n\nprint(\'Report Para Dados de Treino\')\npred_treino = modelo_xgb_v19.predict(dtrain_sc_resample)\npred_treino = pred_treino > 0.5\n\n# Acurácia\nprint("Acurácia: %f" % accuracy_score(y_train_resample_2, pred_treino))\n\n# Classification Report\nprint(classification_report(y_train_resample_2, pred_treino, target_names= [\'VIVER\', \'MORRER\']))  ')


# In[205]:


get_ipython().run_cell_magic('time', '', '\n#Transformando os dados em DMatrix pois o XGBoost exige\ndtrain_sc_resample = xgb.DMatrix(x_train_sc_resample, label = y_train_resample_2)\ndtest_sc = xgb.DMatrix(x_test_sc, label = y_test_nothing)\n\nparams = {\'colsample_bynode\': 0.6, \'colsample_bytree\': 0.5, \'gamma\': 2.2, \'learning_rate\': 0.5, \'max_depth\': 5,\\\n          \'min_child_weight\': 1, \'nthread\': 2, \'subsample\': 0.789, \'colsample_bylevel\': 0.9}\n\n# Criação do modelo base v20\n# Criando sem feature selecting. Com padronização e balanceamento\nmodelo_xgb_v20 = xgb.train(params = params, dtrain = dtrain_sc_resample)\n\npred_v20 = modelo_xgb_v20.predict(dtest_sc)\n\nreport_modelo(modelo_xgb_v20, y_test_nothing, pred_v20, label = \'XGB V20\', target_names = [\'VIVER\', \'MORRER\'])\n\nprint(\'Report Para Dados de Treino\')\npred_treino = modelo_xgb_v20.predict(dtrain_sc_resample)\npred_treino = pred_treino > 0.5\n\n# Acurácia\nprint("Acurácia: %f" % accuracy_score(y_train_resample_2, pred_treino))\n\n# Classification Report\nprint(classification_report(y_train_resample_2, pred_treino, target_names= [\'VIVER\', \'MORRER\']))  ')


# # 7. Conclusão

# Após uma vasto trabalho no problema, conseguimnos atingir um Recall que o problema. O objetivo era um Recall de 75% para a classe 0 (VIVER) e 85% para a classe 1 (MORRER), atingimos 75% e 86%, respectivamente. Ao longo do notebook foram encontrado alguns problemas com o dataset, como por exemplo o seu tamanho que é muito pequeno, 299 observações, desbalanceamento entre as classes, algumas features de baixissima qualidade...
# 
# Porém com um alto processamento nos dados, e uma modelagem intensa sobre os algoritmos de Machine Learning conseguimos reduzir o overfitting e alcançar um Recall desejado. É importante ressaltar que todos os parâmetros abaixo foram testados intensamente, tentando explorar ao maximo o espaço de hipotese.
# 
# Ao longo do treinamento foram utilizados 3 algoritmos, primeiro iniciamos vendo a disposição dos dados com o algoritmo K-Means, logo após iniciamos a modelagem com o algoritmo SVM, onde tivemos muitas dificuldades devido ao numero de observações. Assim, partimos para um algoritmo que lida melhor com menores quantidades de dados o XGBoost, onde com alta modelagem dos hiperparametros atingimos as métricas desejadas.

# Abaixo segue as especificações utilizadas para o resultado final:
# 
# - Feature Selecting: Não. Todas variaveis utilizadas.
# - Padronização: Sim. Somente nas variaveis independentes.
# - Balanceamento (SMOTE): Sim. Somente no dataset de treino.
# 
# 
# - Algoritmo: XGBoost
# - Parâmetros: {'colsample_bynode': 0.6, 'colsample_bytree': 0.5, 'gamma': 2.2, 'learning_rate': 0.5, 'max_depth': 5, 'min_child_weight': 1, 'nthread': 2, 'subsample': 0.789, 'colsample_bylevel': 0.9}

# Abaixo segue o treinamento do modelo final e suas ultimas interpretações:

# In[163]:


# Transformando os dados em DMatrix pois o XGBoost exige
dtrain_sc_resample = xgb.DMatrix(x_train_sc_resample, label = y_train_resample_2)
dtest_sc = xgb.DMatrix(x_test_sc, label = y_test_nothing)

params = {'colsample_bynode': 0.6, 'colsample_bytree': 0.5, 'gamma': 2.2, 'learning_rate': 0.5, 'max_depth': 5,          'min_child_weight': 1, 'nthread': 2, 'subsample': 0.789, 'colsample_bylevel': 0.9}

# Criação do modelo base final
# Criando sem feature selecting. Com padronização e balanceamento
modelo_xgb_final = xgb.train(params = params, dtrain = dtrain_sc_resample)

pred_final = modelo_xgb_final.predict(dtest_sc)

report_modelo(modelo_xgb_final, y_test_nothing, pred_final, label = 'XGB Final', target_names = ['VIVER', 'MORRER'])

print('Report Para Dados de Treino')
pred_treino = modelo_xgb_final.predict(dtrain_sc_resample)
pred_treino = pred_treino > 0.5

# Acurácia
print("Acurácia: %f" % accuracy_score(y_train_resample_2, pred_treino))

# Classification Report
print(classification_report(y_train_resample_2, pred_treino, target_names= ['VIVER', 'MORRER']))  


# In[164]:


colunas = ['age',
         'anaemia',
         'creatinine_phosphokinase',
         'diabetes',
         'ejection_fraction',
         'high_blood_pressure',
         'platelets',
         'serum_creatinine',
         'serum_sodium',
         'sex',
         'smoking']


# In[165]:


shap.initjs()
explainer = shap.TreeExplainer(modelo_xgb_final)
shap_values = explainer(X)


# In[166]:


# Interpretação da predição 0
shap.plots.waterfall(shap_values[0])


# In[167]:


# Interpretação da predição 0
shap.plots.force(shap_values[0])


# In[168]:


# Interpretação da predição 1
shap.plots.waterfall(shap_values[1])


# In[169]:


# Interpretação da predição 1
shap.plots.force(shap_values[1])


# In[170]:


shap.plots.beeswarm(shap_values)


# In[171]:


shap.plots.bar(shap_values)


# In[ ]:





# In[ ]:




