#!/usr/bin/env python
# coding: utf-8

# # Projeto Ciência de Dados - Previsão de Preços
# 
# - Nosso desafio é conseguir prever o preço de barcos que vamos vender baseado nas características do barco, como: ano, tamanho, tipo de barco, se é novo ou usado, qual material usado, etc.
# 
# - Base de Dados: https://drive.google.com/drive/folders/1o2lpxoi9heyQV1hIlsHXWSfDkBPtze-V?usp=share_link

# ### Passo a Passo de um Projeto de Ciência de Dados
# 
# - Passo 1: Entendimento do Desafio
# - Passo 2: Entendimento da Área/Empresa
# - Passo 3: Extração/Obtenção de Dados
# - Passo 4: Ajuste de Dados (Tratamento/Limpeza)
# - Passo 5: Análise Exploratória
# - Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)
# - Passo 7: Interpretação de Resultados

# ![title](tabelas.png)

# In[10]:


#Passo 1: Entendimento do Desafio
#Passo 2: Entendimento da Área/Empresa
    #Prever o preço de um barco baseado nas características dele: ano, material, tipo...
#Passo 3: Extração/Obtenção de Dados

import pandas as pd

tabela = pd.read_csv("barcos_ref.csv")

display(tabela)


# In[11]:


#Passo 4: Ajuste de Dados (Tratamento/Limpeza)
print(tabela.info())


# In[12]:


#Passo 5: Análise Exploratória
correlação = tabela.corr()[["Preco"]]
display(correlação)

import seaborn as sns
import matplotlib.pyplot as plt

#Criar grafico
sns.heatmap(correlação, cmap="Blues",annot= True)
#Exibir grafico
plt.show()


# In[13]:


#Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)

y = tabela["Preco"]
x = tabela.drop("Preco", axis=1)

# train test split
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)


# In[14]:


#Importar IA
    # Regressão linear, e árvore de decisão
    
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

#Criar a IA

modelo_regressaolinear = LinearRegression()

modelo_arvoredecisao = RandomForestRegressor()

#Treinar a IA

modelo_regressaolinear.fit(x_treino, y_treino)

modelo_arvoredecisao.fit(x_treino, y_treino)


# In[16]:


#Passo 7: Interpretação de Resultados 

#Escolher melhor modelo --> R²

from sklearn.metrics import r2_score

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)

previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

print(r2_score(y_teste, previsao_regressaolinear))

print(r2_score(y_teste, previsao_arvoredecisao))


# In[24]:


#Visualizar as previsões


#Fazer novas previsões

tabela_nova = pd.read_csv("novos_barcos.csv")
display(tabela_nova)

previsao = modelo_arvoredecisao.predict(tabela_nova)
print(previsao)


# In[ ]:




