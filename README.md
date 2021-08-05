As doenças cardiovasculares hoje são a maior causadora de mortes no mundo, estimado que 31% das mortes no mundo estão relacionadas a doenças cardiovasculares. Essas doenças causam diretamente a falha do coração que levam a morte do individuo.

A maioria das doenças cardiovasculares podem ser evitadas com pequenos cuidados, porém é dificil saber se algum individuo irá possuir uma doença cardiovascular, dessa forma Machine Learning é de grande utilidade para auxiliar na prevenção dessas doenças.

O objetivo desse projeto é a previsão se o paciente irá ser levado a uma falha cardiovascular pelo desenvolvimento de alguma doença.

 -- Objetivos

  É esperado um Recall minimo de 75% para a classe 0 (DEATH_EVENT não ocorreu). Já para a classe 1 (DEATH_EVENT ocorreu) é esperado um Recall de 85%. 

  A escolha dos objetivos é baseado em algumas analises iniciais, sabendo que o nosso dataset não possui muitas observa-ções e esta desbalanceado. Dessa forma irá precisar de uma modelagem mais cuidadosa para atingir um modelo ideal. 

  Também foi escolhido que iremos priorizar que o algoritmo acerte quem irá MORRER. Dessa forma poderia ser iniciado um tratamento para o paciente antecipadamente.

Dataset: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

Orientações:

- Dentro da pasta docs esta localizado o notebook utilizado durante o projeto, juntamente com a sua versão convertido para .html .
- O script .py na raiz do projeto é uma conversão direta do notebook utilizado, sendo assim é sugerido a utilização do notebook na pasta docs.
