import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.optim import Adam
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product 
import statsmodels.api as sm

# Desativa o aviso PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Tech Challenge - Fase 4")

    # Criando as abas
    tabs = ["Introdução", "Preparação dos Dados", "Exploração dos Dados", "Modelo Machine Learning - LSTM", "Modelo Machine Learning - ARIMA"]
    selected_tab = st.sidebar.radio("Etapas do projeto:", tabs)

    if selected_tab == "Introdução":
        apresentacao_introducao()
    elif selected_tab == "Preparação dos Dados":
        apresentacao_Preparação_dos_Dados()
    elif selected_tab == "Exploração dos Dados":
        apresentacao_Exploração_dos_Dados()
    elif selected_tab == "Modelo Machine Learning - LSTM":
        apresentacao_Modelo_Machine_Learning_LSTM()
    elif selected_tab == "Modelo Machine Learning - ARIMA":
        apresentacao_Modelo_Machine_Learning_ARIMA()

    # Renderiza o link no rodapé da página
    st.markdown("[Visualizar Dashboard no Tableau](https://public.tableau.com/app/profile/lana.morgado.martinez/viz/Apresentao-TechChallenge4/Apresentao-TechChallenge4?publish=yes)")


def apresentacao_introducao():
    st.header("Introdução")
    st.write("Bem-vindo ao nosso dashboard interativo sobre análise e previsão do preço do petróleo Brent! Este projeto faz parte do nosso trabalho para a pós-graduação em Data Analytics, no qual nos propusemos a abordar três aspectos cruciais:")
    st.write("**Análise do Mercado de Petróleo:** Utilizando dados históricos do preço do petróleo fornecidos pelo IPEA, fizemos uma **Exploração e Preparação de Dados** onde identificamos tendências, padrões e influências que moldam o mercado. Vamos destacar quatro insights relevantes sobre variações de preços, eventos geopolíticos, crises econômicas e demanda energética.")
    st.write("**Modelo de Previsão:** Desenvolvemos um modelo de Machine Learning para prever o preço do petróleo diariamente, levando em consideração a natureza temporal dos dados. Você encontrará uma análise detalhada da performance do modelo, incluindo o código utilizado e suas métricas de avaliação.")
    st.write("**Deploy em Produção com Streamlit:** Para tornar nosso modelo acessível e prático, criamos um MVP utilizando o Streamlit. Este MVP representa um passo crucial em direção à implementação prática de soluções analíticas em um ambiente de produção.")
    st.write("Nossa jornada não se limita apenas a entender o passado, mas também a antecipar o futuro. Estamos entusiasmados em compartilhar nossas descobertas e insights com você. Vamos explorar juntos!")

def apresentacao_Preparação_dos_Dados():
    st.header("Exploração e Preparação de Dados")

    # Carregar o arquivo Excel
    df = pd.read_excel('Petróleo.xlsx')

    # Adicione seu código aqui
    codigo = """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    import calendar

    # Carregar o arquivo Excel
    df = pd.read_excel('Petróleo.xlsx')

    print(df.head())

    # Verificar a quantidade de dados
    print("Quantidade de dados antes do tratamento:")
    print(df.shape)

    # Tratar valores ausentes (remover neste exemplo)
    df = df.dropna()

    # Verificar a quantidade de dados após tratamento de valores ausentes
    print("\nQuantidade de dados após tratamento de valores ausentes:")
    print(df.shape)

    # Verificar inconsistências (por exemplo, datas fora do intervalo esperado)
    # Suponha que a data deve estar entre 2000 e 2024
    inconsistencias = df[(df['Data'].dt.year < 2000) | (df['Data'].dt.year > 2024)]

    # Remover inconsistências (ou tratar de outra forma)
    df = df.drop(inconsistencias.index)

    # Identificar e tratar outliers (exemplo usando IQR)
    Q1 = df['Preço - petróleo bruto - Brent (FOB)'].quantile(0.25)
    Q3 = df['Preço - petróleo bruto - Brent (FOB)'].quantile(0.75)
    IQR = Q3 - Q1

    # Definir limites para outliers
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    # Identificar outliers
    outliers = df[(df['Preço - petróleo bruto - Brent (FOB)'] < limite_inferior) | (df['Preço - petróleo bruto - Brent (FOB)'] > limite_superior)]

    # Remover outliers (ou tratar de outra forma)
    df = df[~df.index.isin(outliers.index)]

    # Verificar a quantidade de dados após tratamento de outliers
    print("\nQuantidade de dados após tratamento de outliers:")
    print(df.shape)

    # Verificar tipos de dados das colunas
    tipos_de_dados = df.dtypes

    # Exibir tipos de dados das colunas
    print("Tipos de dados das colunas:")
    print(tipos_de_dados)
    """

    # Apresentar o código
    st.subheader("Código:")
    st.code(codigo, language='python')

    # Exibir as primeiras linhas do dataframe
    st.subheader("Visualização das primeiras linhas do DataFrame:")
    st.dataframe(df.head())

    # Verificar a quantidade de dados
    st.subheader("Quantidade de dados antes do tratamento:")
    st.write(df.shape)

    # Tratar valores ausentes (remover neste exemplo)
    df = df.dropna()

    # Exibir a quantidade de dados após o tratamento de valores ausentes
    st.subheader("Quantidade de dados após o tratamento de valores ausentes:")
    st.write(df.shape)

    # Verificar inconsistências (por exemplo, datas fora do intervalo esperado)
    # Suponha que a data deve estar entre 2000 e 2024
    inconsistencias = df[(df['Data'].dt.year < 2000) | (df['Data'].dt.year > 2024)]

    # Remover inconsistências (ou tratar de outra forma)
    df = df.drop(inconsistencias.index)

    # Identificar e tratar outliers (exemplo usando IQR)
    Q1 = df['Preço - petróleo bruto - Brent (FOB)'].quantile(0.25)
    Q3 = df['Preço - petróleo bruto - Brent (FOB)'].quantile(0.75)
    IQR = Q3 - Q1

    # Definir limites para outliers
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    # Identificar outliers
    outliers = df[(df['Preço - petróleo bruto - Brent (FOB)'] < limite_inferior) | (df['Preço - petróleo bruto - Brent (FOB)'] > limite_superior)]

    # Remover outliers (ou tratar de outra forma)
    df = df[~df.index.isin(outliers.index)]

    # Exibir a quantidade de dados após o tratamento de outliers
    st.subheader("Quantidade de dados após tratamento de inconsistências e outliers:")
    st.write(df.shape)

    # Verificar tipos de dados das colunas
    tipos_de_dados = df.dtypes

    # Exibir tipos de dados das colunas
    st.subheader("Tipos de dados das colunas:")
    st.write(tipos_de_dados)

    # Texto adicional
    st.subheader("Visões Interessantes:")
    st.write("""
    Com os dados tratados e entendendo que temos informações sobre o preço do petróleo bruto Brent (FOB) em dólares americanos ao longo do tempo, podemos extrair diversas análises e visualizações interessantes para compreender melhor o comportamento desse mercado. Algumas visões interessantes incluem:

    1 - Tendência ao longo do tempo: Visualizar como o preço do petróleo Brent tem variado ao longo dos anos.

    2 - Padrões sazonais: Investigar se existem padrões sazonais nos preços do petróleo ao longo dos meses ou trimestres.

    3 - Correlações: Verificar se há correlações entre os preços do petróleo e outros indicadores econômicos, como o preço do dólar, o PIB mundial, entre outros.

    4 - Análise de volatilidade: Avaliar a volatilidade dos preços ao longo do tempo e identificar períodos de alta volatilidade.
    """)

    # Botão para ir para a aba "Exploração"
    if st.button("Continuar para Exploração dos dados"):
        apresentacao_Exploração_dos_Dados()

    return df

def apresentacao_Exploração_dos_Dados():
    st.header("Exploração dos dados")

    # Adicione seu código aqui
    codigo = """
    # Visualizar tendência ao longo do tempo
    plt.figure(figsize=(12, 6))
    plt.plot(df['Data'], df['Preço - petróleo bruto - Brent (FOB)'])
    plt.title('Preço do Petróleo Brent ao Longo do Tempo')
    plt.xlabel('Data')
    plt.ylabel('Preço em Dólares')
    plt.grid(True)
    plt.show()

    # Verificar padrões sazonais (por exemplo, por mês)
    df['Mês'] = df['Data'].dt.month
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Mês', y='Preço - petróleo bruto - Brent (FOB)', data=df)
    plt.title('Padrões Sazonais no Preço do Petróleo Brent (por mês)')
    plt.xlabel('Mês')
    plt.ylabel('Preço em Dólares')
    plt.grid(True)
    plt.show()

    #Tratando a nova planilha de correlação com o dolar
    df_dolar = pd.read_csv('USD_BRL_Dados_Históricos.csv', delimiter=';')

    # Converter a coluna 'Data' para o mesmo formato em ambos os DataFrames
    df['Data'] = pd.to_datetime(df['Data'])
    df_dolar['Data'] = pd.to_datetime(df_dolar['Data'])

    # Renomear coluna 'Último' para 'Valor Real Dólar' para facilitar a compreensão
    df_dolar = df_dolar.rename(columns={'Último': 'Valor Real Dólar'})

    # Mesclar os DataFrames pelo índice (Data)
    df_merged = pd.merge(df, df_dolar, on='Data', how='inner')

    # Verificar tipos de dados das colunas
    print(df_merged.dtypes)

    # Verificar se há valores não numéricos
    print(df_merged['Preço - petróleo bruto - Brent (FOB)'].unique())
    print(df_merged['Valor Real Dólar'].unique())

    # Carregar os arquivos do CSV da planilha dollar
    df = pd.read_excel('Petróleo.xlsx')
    df_dolar = pd.read_csv('USD_BRL_Dados_Históricos.csv', delimiter=';')

    # Converter a coluna 'Data' para o mesmo formato em ambos os DataFrames
    df['Data'] = pd.to_datetime(df['Data'])
    df_dolar['Data'] = pd.to_datetime(df_dolar['Data'])

    # Renomear coluna 'Último' para 'Valor Real Dólar' para facilitar a compreensão
    df_dolar = df_dolar.rename(columns={'Último': 'Valor Real Dólar'})

    # Substituir vírgulas por pontos e converter para tipo float
    df_dolar['Valor Real Dólar'] = df_dolar['Valor Real Dólar'].str.replace(',', '.').astype(float)

    # Mesclar os DataFrames pelo índice (Data)
    df_merged = pd.merge(df, df_dolar, on='Data', how='inner')

    # Calcular a correlação entre o preço do petróleo Brent e o valor real do dólar
    correlacao = df_merged['Preço - petróleo bruto - Brent (FOB)'].corr(df_merged['Valor Real Dólar'])

    print("Correlação entre preço do petróleo Brent e valor real do dólar:", correlacao)

    # Mesclar os DataFrames pelo índice (Data)
    df_merged = pd.merge(df, df_dolar, on='Data', how='inner')

    # Criar um gráfico de dispersão
    plt.figure(figsize=(10, 6))
    plt.scatter(df_merged['Preço - petróleo bruto - Brent (FOB)'], df_merged['Valor Real Dólar'], alpha=0.7)
    plt.title('Correlação entre Preço do Petróleo Brent e Valor Real do Dólar')
    plt.xlabel('Preço do Petróleo Brent (USD)')
    plt.ylabel('Valor Real do Dólar (BRL)')
    plt.grid(True)
    plt.show()

    # Converter a coluna 'Data' para datetime
    df['Data'] = pd.to_datetime(df['Data'])

    # Filtrar os dados para o intervalo de 2014 a 2024
    df_intervalo = df[(df['Data'] >= '2014-01-01') & (df['Data'] <= '2024-12-31')]

    # Encontrar o maior e o menor valor do preço do petróleo e os períodos correspondentes
    maior_valor = df_intervalo['Preço - petróleo bruto - Brent (FOB)'].max()
    menor_valor = df_intervalo['Preço - petróleo bruto - Brent (FOB)'].min()

    periodo_maior_valor = df_intervalo.loc[df_intervalo['Preço - petróleo bruto - Brent (FOB)'] == maior_valor, 'Data'].iloc[0]
    periodo_menor_valor = df_intervalo.loc[df_intervalo['Preço - petróleo bruto - Brent (FOB)'] == menor_valor, 'Data'].iloc[0]

    # Plotar o gráfico de linha para visualizar a variação do preço ao longo do tempo
    plt.figure(figsize=(10, 6))
    plt.plot(df_intervalo['Data'], df_intervalo['Preço - petróleo bruto - Brent (FOB)'], color='blue', linewidth=1)
    plt.title('Variação do Preço do Petróleo Brent (2014-2024)')
    plt.xlabel('Data')
    plt.ylabel('Preço do Petróleo Brent (USD)')
    plt.grid(True)

    # Adicionar pontos para destacar o maior e o menor valor
    plt.scatter(periodo_maior_valor, maior_valor, color='red', label=f'Maior Valor: {maior_valor} em {periodo_maior_valor.strftime("%Y-%m-%d")}')
    plt.scatter(periodo_menor_valor, menor_valor, color='green', label=f'Menor Valor: {menor_valor} em {periodo_menor_valor.strftime("%Y-%m-%d")}')

    plt.legend()
    plt.show()

    # Converter a coluna 'Data' para datetime
    df['Data'] = pd.to_datetime(df['Data'])

    # Filtrar os dados para o intervalo de 2014 a 2024
    df_intervalo = df[(df['Data'] >= '2014-01-01') & (df['Data'] <= '2024-12-31')]

    # Encontrar o maior e o menor valor do preço do petróleo e os períodos correspondentes
    maior_valor = df_intervalo['Preço - petróleo bruto - Brent (FOB)'].max()
    menor_valor = df_intervalo['Preço - petróleo bruto - Brent (FOB)'].min()

    periodo_maior_valor = df_intervalo.loc[df_intervalo['Preço - petróleo bruto - Brent (FOB)'] == maior_valor, 'Data'].iloc[0]
    periodo_menor_valor = df_intervalo.loc[df_intervalo['Preço - petróleo bruto - Brent (FOB)'] == menor_valor, 'Data'].iloc[0]

    # Plotar o gráfico de linha para visualizar a variação do preço ao longo do tempo
    plt.figure(figsize=(10, 6))
    plt.plot(df_intervalo['Data'], df_intervalo['Preço - petróleo bruto - Brent (FOB)'], color='blue', linewidth=1)
    plt.title('Variação do Preço do Petróleo Brent (2014-2024)')
    plt.xlabel('Data')
    plt.ylabel('Preço do Petróleo Brent (USD)')
    plt.grid(True)

    # Adicionar pontos para destacar o maior e o menor valor
    plt.scatter(periodo_maior_valor, maior_valor, color='red', label=f'Maior Valor: {maior_valor} em {periodo_maior_valor.strftime("%Y-%m-%d")}')
    plt.scatter(periodo_menor_valor, menor_valor, color='green', label=f'Menor Valor: {menor_valor} em {periodo_menor_valor.strftime("%Y-%m-%d")}')

    plt.legend()
    plt.show()

    # Calcular os retornos diários do preço do petróleo
    df['Retornos'] = df['Preço - petróleo bruto - Brent (FOB)'].pct_change()

    # Calcular o desvio padrão dos retornos em janelas móveis de 30 dias
    window = 30
    df['Volatilidade'] = df['Retornos'].rolling(window).std() * (252**0.5)  # Fator de ajuste para considerar dias úteis em um ano

    # Plotar a volatilidade ao longo do tempo
    plt.figure(figsize=(10, 6))
    plt.plot(df['Data'], df['Volatilidade'], color='blue', linewidth=1)
    plt.title('Volatilidade do Preço do Petróleo Brent')
    plt.xlabel('Data')
    plt.ylabel('Volatilidade (Anualizada)')
    plt.grid(True)
    plt.show()

    # Calcular os retornos diários do preço do petróleo
    df['Retornos'] = df['Preço - petróleo bruto - Brent (FOB)'].pct_change()

    # Calcular o desvio padrão dos retornos em janelas móveis de 30 dias
    window = 30
    df['Volatilidade'] = df['Retornos'].rolling(window).std() * (252**0.5)  # Fator de ajuste para considerar dias úteis em um ano

    # Extrair o ano de cada data
    df['Ano'] = df['Data'].dt.year

    # Calcular a volatilidade média de cada ano
    df_volatilidade_ano = df.groupby('Ano')['Volatilidade'].mean().reset_index()

    # Calcular a variação percentual da volatilidade ano a ano
    df_volatilidade_ano['Variação Percentual'] = df_volatilidade_ano['Volatilidade'].pct_change() * 100

    # Plotar a variação percentual ano a ano da volatilidade
    plt.figure(figsize=(10, 6))
    plt.bar(df_volatilidade_ano['Ano'], df_volatilidade_ano['Variação Percentual'], color='blue')
    plt.title('Variação Percentual Ano a Ano da Volatilidade do Preço do Petróleo Brent')
    plt.xlabel('Ano')
    plt.ylabel('Variação Percentual (%)')
    plt.grid(True)
    plt.xticks(df_volatilidade_ano['Ano'])  # Exibir todos os anos no eixo x
    plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Linha horizontal para representar a variação zero
    plt.show()

    # Extrair o ano de cada data
    df['Ano'] = df['Data'].dt.year

    # Plotar uma linha para cada ano
    plt.figure(figsize=(10, 6))
    for ano in df['Ano'].unique():
        df_ano = df[df['Ano'] == ano]
        plt.plot(df_ano['Data'], df_ano['Preço - petróleo bruto - Brent (FOB)'], label=ano)

    plt.title('Preço do Petróleo Brent Ano a Ano')
    plt.xlabel('Data')
    plt.ylabel('Preço do Petróleo Brent')
    plt.grid(True)
    plt.legend(title='Ano')
    plt.show()
    """
    # Apresentar o código
    st.subheader("Código:")
    st.code(codigo, language='python')

    # Carregar o arquivo Excel
    df = pd.read_excel('Petróleo.xlsx')

    # Tratando a nova planilha de correlação com o dólar
    df_dolar = pd.read_csv('USD_BRL Dados Históricos.csv', delimiter=',', thousands='.', decimal=',', parse_dates=['Data'], dayfirst=True)

    # Converter a coluna 'Data' para o mesmo formato em ambos os DataFrames
    df['Data'] = pd.to_datetime(df['Data'])
    df_dolar = df_dolar.rename(columns={'Último': 'Valor Real Dólar'})

    # Renomear coluna 'Último' para 'Valor Real Dólar' para facilitar a compreensão
    df_dolar = df_dolar.rename(columns={'Último': 'Valor Real Dólar'})

    # Mesclar os DataFrames pelo índice (Data)
    df_merged = pd.merge(df, df_dolar, on='Data', how='inner')

    # Visualizar tendência ao longo do tempo
    st.subheader("Tendência ao Longo do Tempo")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Data'], df['Preço - petróleo bruto - Brent (FOB)'])
    ax.set_title('Preço do Petróleo Brent ao Longo do Tempo')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço em Dólares')
    ax.grid(True)
    st.pyplot(fig)

    # Verificar padrões sazonais (por exemplo, por mês)
    st.subheader("Padrões sazonais por mês")
    df['Mês'] = df['Data'].dt.month
    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.boxplot(x='Mês', y='Preço - petróleo bruto - Brent (FOB)', data=df)
    ax.set_title('Padrões Sazonais no Preço do Petróleo Brent (por mês)')
    ax.set_xlabel('Mês')
    ax.set_ylabel('Preço em Dólares')
    ax.grid(True)
    st.pyplot(fig)

    # Exibir tipos de dados das colunas e valores únicos
    st.subheader("Tipos de Dados e Valores Únicos:")
    st.write(df_merged.dtypes)
    st.write("Valores únicos do preço do petróleo Brent:", df_merged['Preço - petróleo bruto - Brent (FOB)'].unique())
    st.write("Valores únicos do valor real do dólar:", df_merged['Valor Real Dólar'].unique())

    # Converter a coluna 'Data' para o mesmo formato em ambos os DataFrames
    df['Data'] = pd.to_datetime(df['Data'])
    df_dolar['Data'] = pd.to_datetime(df_dolar['Data'])

    # Renomear coluna 'Último' para 'Valor Real Dólar' para facilitar a compreensão
    df_dolar = df_dolar.rename(columns={'Último': 'Valor Real Dólar'})

    # Substituir vírgulas por pontos e converter para tipo float
    df_dolar['Valor Real Dólar'] = df_dolar['Valor Real Dólar'].astype(str).str.replace(',', '.').astype(float)

    # Mesclar os DataFrames pelo índice (Data)
    df_merged = pd.merge(df, df_dolar, on='Data', how='inner')

    # Calcular a correlação entre o preço do petróleo Brent e o valor real do dólar
    correlacao = df_merged['Preço - petróleo bruto - Brent (FOB)'].corr(df_merged['Valor Real Dólar'])

    st.write("Correlação entre preço do petróleo Brent e valor real do dólar: 0.2875.")
    st.write("Essa correlação positiva sugere que há uma relação fraca a moderada entre essas duas variáveis. Um valor positivo indica que, em geral, quando o preço do petróleo Brent aumenta, o valor real do dólar também tende a aumentar, e vice-versa. No entanto, como a correlação não é muito alta (próxima de 0.3), a relação entre essas variáveis pode não ser muito forte, e outros fatores também podem estar influenciando seus movimentos.")
    st.write("Essa análise pode ser útil para entender como essas duas variáveis estão inter-relacionadas e pode ter implicações para investidores ou empresas que operam em setores afetados pelos preços do petróleo e pela taxa de câmbio.")

    # Criar um gráfico de dispersão
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_merged['Preço - petróleo bruto - Brent (FOB)'], df_merged['Valor Real Dólar'], alpha=0.7)
    ax.set_title('Correlação entre Preço do Petróleo Brent e Valor Real do Dólar')
    ax.set_xlabel('Preço do Petróleo Brent (USD)')
    ax.set_ylabel('Valor Real do Dólar (BRL)')
    ax.grid(True)
    st.pyplot()

    st.write("Ausência de Correlação: Se os pontos estão distribuídos aleatoriamente no gráfico, sem uma tendência clara, isso sugere que não há correlação significativa entre as duas variáveis.")


    # Converter a coluna 'Data' para datetime
    df['Data'] = pd.to_datetime(df['Data'])

    # Filtrar os dados para o intervalo de 2014 a 2024
    df_intervalo = df[(df['Data'] >= '2014-01-01') & (df['Data'] <= '2024-12-31')]

    # Encontrar o maior e o menor valor do preço do petróleo e os períodos correspondentes
    maior_valor = df_intervalo['Preço - petróleo bruto - Brent (FOB)'].max()
    menor_valor = df_intervalo['Preço - petróleo bruto - Brent (FOB)'].min()

    periodo_maior_valor = df_intervalo.loc[df_intervalo['Preço - petróleo bruto - Brent (FOB)'] == maior_valor, 'Data'].iloc[0]
    periodo_menor_valor = df_intervalo.loc[df_intervalo['Preço - petróleo bruto - Brent (FOB)'] == menor_valor, 'Data'].iloc[0]

    # Criar um gráfico de linha para visualizar a variação do preço ao longo do tempo
    st.subheader('Variação do Preço do Petróleo Brent (2014-2024)')
    st.line_chart(df_intervalo.set_index('Data')['Preço - petróleo bruto - Brent (FOB)'])

    # Criar um gráfico de dispersão para destacar o maior e o menor valor
    st.subheader('Maior e Menor Valor do Preço do Petróleo Brent')
    fig, ax = plt.subplots()
    ax.plot(df_intervalo['Data'], df_intervalo['Preço - petróleo bruto - Brent (FOB)'], color='blue', linewidth=1)
    ax.scatter(periodo_maior_valor, maior_valor, color='red', label=f'Maior Valor: {maior_valor} em {periodo_maior_valor.strftime("%Y-%m-%d")}')
    ax.scatter(periodo_menor_valor, menor_valor, color='green', label=f'Menor Valor: {menor_valor} em {periodo_menor_valor.strftime("%Y-%m-%d")}')
    ax.legend()
    st.pyplot(fig)

    st.write("**2020** - Em 2020, o mercado de petróleo enfrentou uma série de desafios significativos que contribuíram para a queda dos preços do petróleo Brent, atingindo um valor mínimo em abril daquele ano. Alguns dos principais fatores que influenciaram os preços do petróleo em 2020 incluem: Pandemia de COVID-19: A pandemia de COVID-19 teve um impacto dramático na demanda global por petróleo. Com as medidas de lockdown e restrições de viagem implementadas em todo o mundo para conter a propagação do vírus, houve uma redução acentuada na atividade econômica, levando a uma queda na demanda por combustíveis, como gasolina, diesel e querosene de aviação. Guerra de Preços entre Arábia Saudita e Rússia: Em março de 2020, a Arábia Saudita e a Rússia entraram em uma guerra de preços do petróleo, aumentando sua produção e inundando o mercado com petróleo em meio a uma demanda já enfraquecida pela pandemia. Isso levou a uma queda significativa nos preços do petróleo. Excesso de Oferta e Armazenamento Limitado: Com o aumento da produção e a queda na demanda, houve um excesso de oferta de petróleo no mercado global. Isso levou a uma saturação dos estoques de petróleo e à falta de capacidade de armazenamento, resultando em preços negativos para os contratos futuros de petróleo pela primeira vez na história. Interrupções na Produção: Além dos fatores mencionados, também houve interrupções na produção de petróleo devido a eventos climáticos extremos e conflitos geopolíticos, contribuindo para a volatilidade e incerteza nos mercados de commodities. Em resumo, a combinação da queda na demanda devido à pandemia de COVID-19, o excesso de oferta resultante da guerra de preços entre Arábia Saudita e Rússia e as restrições de armazenamento devido à saturação do mercado contribuíram para os preços historicamente baixos do petróleo Brent em abril de 2020.")
    st.write("**2022** - Em 2022, diversos fatores podem ter contribuído para o aumento dos preços do petróleo Brent, embora o cenário exato possa ser complexo e multifacetado. Aqui estão algumas possíveis justificativas para o aumento dos preços do petróleo em 2022: Recuperação Econômica pós-COVID-19: Com a implementação de programas de vacinação e a redução das restrições relacionadas à pandemia de COVID-19 em muitas partes do mundo, a atividade econômica começou a se recuperar. Isso pode ter levado a um aumento na demanda por energia, incluindo petróleo, à medida que as indústrias retomaram suas operações e o consumo de combustível aumentou. Restrições na Oferta de Petróleo: Em resposta à queda nos preços do petróleo em 2020 e 2021, muitos países produtores de petróleo reduziram sua produção para equilibrar o mercado e sustentar os preços. Essas restrições na oferta podem ter ajudado a impulsionar os preços do petróleo em 2022. Tensões Geopolíticas: Eventos geopolíticos, como tensões no Oriente Médio ou instabilidade política em países produtores de petróleo, podem ter gerado preocupações quanto à interrupção da oferta de petróleo. Essas preocupações podem ter contribuído para a elevação dos preços do petróleo. Recuperação da Demanda Global por Energia: Com a retomada da atividade econômica e o aumento da demanda por energia em todo o mundo, especialmente em setores como transporte, manufatura e aviação, a demanda por petróleo pode ter aumentado, impulsionando os preços. Inflação e Pressões Inflacionárias: A inflação global e as preocupações com pressões inflacionárias podem ter levado os investidores a buscar ativos de commodities, incluindo o petróleo, como uma proteção contra a desvalorização da moeda e a perda de poder de compra. É importante ressaltar que o mercado de petróleo é influenciado por uma variedade de fatores inter-relacionados e que a dinâmica dos preços pode ser complexa e sujeita a mudanças rápidas. Assim, enquanto esses fatores podem ter contribuído para o aumento dos preços do petróleo em 2022, outros eventos e condições também podem ter desempenhado um papel significativo.")

    # Calcular os retornos diários do preço do petróleo
    df['Retornos'] = df['Preço - petróleo bruto - Brent (FOB)'].pct_change()

    # Calcular o desvio padrão dos retornos em janelas móveis de 30 dias
    window = 30
    df['Volatilidade'] = df['Retornos'].rolling(window).std() * (252**0.5)  # Fator de ajuste para considerar dias úteis em um ano

    # Plotar a volatilidade ao longo do tempo
    st.subheader('Volatilidade do Preço do Petróleo Brent')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Data'], df['Volatilidade'], color='blue', linewidth=1)
    ax.set_title('Volatilidade do Preço do Petróleo Brent')
    ax.set_xlabel('Data')
    ax.set_ylabel('Volatilidade (Anualizada)')
    ax.grid(True)
    st.pyplot(fig)

    st.write("Calculamos os retornos diários do preço do petróleo Brent usando a função pct_change(). Em seguida, calculamos o desvio padrão dos retornos em janelas móveis de 30 dias para suavizar a volatilidade e torná-la mais facilmente interpretável. Multiplicamos o desvio padrão pelos dias úteis em um ano (no caso, 252) para anualizar a volatilidade. Finalmente, plotamos a volatilidade ao longo do tempo. Este gráfico mostrará como a volatilidade dos preços do petróleo Brent variou ao longo do período de tempo em sua base de dados. Você pode identificar os períodos de maior e menor volatilidade e explorar eventos ou condições que possam ter contribuído para essas mudanças na volatilidade.")

    # Calcular os retornos diários do preço do petróleo
    df['Retornos'] = df['Preço - petróleo bruto - Brent (FOB)'].pct_change()

    # Calcular o desvio padrão dos retornos em janelas móveis de 30 dias
    window = 30
    df['Volatilidade'] = df['Retornos'].rolling(window).std() * (252**0.5)  # Fator de ajuste para considerar dias úteis em um ano

    # Extrair o ano de cada data
    df['Ano'] = df['Data'].dt.year

    # Calcular a volatilidade média de cada ano
    df_volatilidade_ano = df.groupby('Ano')['Volatilidade'].mean().reset_index()

    # Calcular a variação percentual da volatilidade ano a ano
    df_volatilidade_ano['Variação Percentual'] = df_volatilidade_ano['Volatilidade'].pct_change() * 100

    # Plotar a variação percentual ano a ano da volatilidade
    st.subheader('Variação Percentual Ano a Ano da Volatilidade do Preço do Petróleo Brent')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_volatilidade_ano['Ano'], df_volatilidade_ano['Variação Percentual'], color='blue')
    ax.set_title('Variação Percentual Ano a Ano da Volatilidade do Preço do Petróleo Brent')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Variação Percentual (%)')
    ax.grid(True)
    ax.set_xticks(df_volatilidade_ano['Ano'])  # Exibir todos os anos no eixo x
    ax.axhline(0, color='red', linestyle='--', linewidth=1)  # Linha horizontal para representar a variação zero
    st.pyplot(fig)

    # Extrair o ano de cada data
    df['Ano'] = df['Data'].dt.year

    # Plotar uma linha para cada ano
    st.subheader('Preço do Petróleo Brent Ano a Ano')
    fig, ax = plt.subplots(figsize=(10, 6))
    for ano in df['Ano'].unique():
        df_ano = df[df['Ano'] == ano]
        ax.plot(df_ano['Data'], df_ano['Preço - petróleo bruto - Brent (FOB)'], label=str(ano))

    ax.set_title('Preço do Petróleo Brent Ano a Ano')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço do Petróleo Brent')
    ax.grid(True)
    ax.legend(title='Ano')
    st.pyplot(fig)

def apresentacao_Modelo_Machine_Learning_LSTM():
    st.header("Modelo Machine Learning - LSTM")

    # 1
    codigo = """import pandas as pd
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator
    """
    # 1
    st.subheader("Bibliotecas:")
    st.code(codigo, language='python')

    # 2
    codigo = """# Carrega os dados do arquivo Excel
    df = pd.read_excel("Petróleo.xlsx")

    # Mostra as primeiras linhas do DataFrame
    print(df.head())
    """
    # 2
    st.subheader("Código:")
    st.code(codigo, language='python')

    # 3
    codigo = """df
    """
    # 3
    st.subheader("Código:")
    st.code(codigo, language='python')

    # 4
    codigo = """# Função para atualizar o DataFrame com novos dados
    def update_dataframe(df, new_data):
    # Converte a coluna 'Data' para datetime
    df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
    new_data['Data'] = pd.to_datetime(new_data['Data'], dayfirst=True)

    # Encontra a data mais recente no DataFrame existente
    last_date = df['Data'].max()

    # Filtra as novas linhas que são mais recentes do que a última data
    new_rows = new_data[new_data['Data'] > last_date]

    # Concatena os novos dados com o DataFrame existente se houver novas linhas
    if not new_rows.empty:
        updated_df = pd.concat([df, new_rows], ignore_index=True)
    else:
        updated_df = df

    return updated_df

    # Carrega os dados do arquivo Excel
    new_df = pd.read_excel("Petróleo.xlsx")

    # Verifica se o arquivo do DataFrame existe e carrega, ou cria um novo DataFrame se não existir
    path = 'ipea.csv'  # Especifique o nome do arquivo no mesmo diretório do seu script
    try:
        existing_df = pd.read_csv(path)
    except FileNotFoundError:
        existing_df = new_df  # Se o arquivo não existir, considere os dados atuais como o DataFrame existente

    # Atualiza o DataFrame existente com novos dados (carga incremental)
    updated_df = update_dataframe(existing_df, new_df)

    updated_df['Preço - petróleo bruto - Brent (FOB)'] = updated_df['Preço - petróleo bruto - Brent (FOB)']/100

    # Salva o DataFrame atualizado para o arquivo no mesmo diretório do script
    updated_df.to_csv(path, index=False)

    # Mostra as primeiras linhas do DataFrame atualizado
    print(updated_df.head())
    """
    # 4
    st.subheader("Código:")
    st.code(codigo, language='python')
    
    # 5
    codigo = """updated_df.info()
    """
    # 5
    st.subheader("Código:")
    st.code(codigo, language='python')

    # 6
    codigo = """# Definir o dispositivo para a GPU se disponível
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carregar o DataFrame
    df = pd.read_csv('ipea.csv')

    # Converter a coluna de data para datetime
    df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d')

    # Converter a coluna de data para um timestamp Unix
    df['Timestamp'] = df['Data'].apply(lambda x: x.timestamp())

    # Escalar a coluna de preços, já que os modelos de DL geralmente funcionam
    # melhor com dados normalizados
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df['Preço - petróleo bruto - Brent (FOB)'] = scaler.fit_transform(df['Preço - petróleo bruto - Brent (FOB)'].values.reshape(-1, 1)).astype('float32')

    # Preparar dados para o PyTorch
    X = df['Timestamp'].values.astype('float32')  # A entrada do modelo será o timestamp
    y = df['Preço - petróleo bruto - Brent (FOB)'].values.astype('float32')  # A saída do modelo serão os preços

    # Dividir o conjunto de dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Converter os dados para Tensor
    X_train_tensor = torch.tensor(X_train).view(-1, 1, 1)
    y_train_tensor = torch.tensor(y_train).view(-1, 1, 1)
    X_test_tensor = torch.tensor(X_test).view(-1, 1, 1)
    y_test_tensor = torch.tensor(y_test).view(-1, 1, 1)

    # Mover para o dispositivo apropriado
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    # Definir o modelo LSTM
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=200, output_size=1):
            super(LSTMModel, self).__init__()
            self.hidden_layer_size = hidden_layer_size

            self.lstm = nn.LSTM(input_size, hidden_layer_size ,num_layers=3)

            self.linear = nn.Linear(hidden_layer_size, output_size)

        def forward(self, input_seq):
            lstm_out, _ = self.lstm(input_seq)
            predictions = self.linear(lstm_out.view(len(input_seq), -1))
            return predictions[-1]

    # Instanciar o modelo
    model = LSTMModel().to(device)

    # Definir a função de perda e o otimizador
    loss_function = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.0001)

    # Treinar o modelo
    epochs = 10
    for i in range(epochs):
        for seq, labels in zip(X_train_tensor, y_train_tensor):
            optimizer.zero_grad()

            model.hidden = (torch.zeros(3, 1, model.hidden_layer_size).to(device),
                            torch.zeros(3, 1, model.hidden_layer_size).to(device))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

            print(f'Epoch {i} loss: {single_loss.item()}')

    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(len(X_test)):
            seq = X_test_tensor[i : i + 1]
            preds.append(model(seq).cpu().numpy()[0])

    # Inverter a escala dos preços para a escala original
    actual_predictions = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

    # Plotar
    plt.figure(figsize=(15,5))
    plt.plot(df.index[-len(actual_predictions):], actual_predictions, label='Predicted')
    plt.plot(df.index[-len(actual_predictions):], scaler.inverse_transform(y_test.reshape(-1,1)), label='Actual')
    plt.legend()
    plt.show()

    """
    # 6
    st.subheader("Código:")
    st.code(codigo, language='python')

    # 7
    codigo = """# Prever usando o conjunto de teste
    predictions = []
    with torch.no_grad():
        for i in range(len(X_test)):
            seq = X_test_tensor[i : i+1]
            predictions.append(model(seq).item())
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions) # Reverter escala dos dados de previsão

    # Reverter escala dos dados reais de teste
    actual = scaler.inverse_transform(y_test.reshape(-1,1))

    # Plotagem do gráfico
    plt.figure(figsize=(15,6))
    plt.plot(df['Data'].iloc[-len(predictions):], actual, label='Actual Data')
    plt.plot(df['Data'].iloc[-len(predictions):], predictions, label='Predicted Data')
    plt.legend()
    plt.title('Preços reais vs previsões')
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.grid(True)
    plt.show()
    """
    # 7
    st.subheader("Código:")
    st.code(codigo, language='python')

    # 8
    codigo = """
    # Carregar o DataFrame
    df = pd.read_excel('Petróleo.xlsx')
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values(by='Data', ascending=True).reset_index(drop=True)

    # Criar recursos de atraso (lag features) para séries temporais
    for lag in range(1, 2):  # Criar atrasos de 1 dia até 3 dias
        df[f'Preço_lag_{lag}'] = df['Preço - petróleo bruto - Brent (FOB)'].shift(lag)

    # Remover linhas com valores NaN que foram criados ao fazer o shift
    df = df.dropna()

    # Preparar os dados para treinamento
    X = df[['Preço_lag_1']].values  # Inputs são os preços atrasados
    y = df['Preço - petróleo bruto - Brent (FOB)'].values  # Output é o preço atual

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)

    # Criar e treinar o modelo de Gradient Boosting
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, loss='squared_error')
    model.fit(X_train, y_train)

    # Fazer previsões
    predictions = model.predict(X_test)

    # Avaliar o modelo
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Imprimir resultados da avaliação
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)

    # Plotar resultados reais vs previstos
    plt.figure(figsize=(15, 5))
    plt.plot(df['Data'].iloc[-len(y_test):], y_test, label='Real')
    plt.plot(df['Data'].iloc[-len(predictions):], predictions, label='Previsão')

    # Melhorar a formatação do eixo x
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Formatar datas como 'Ano-Mês-Dia'
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())             # Escolher automaticamente a localização das datas
    plt.gcf().autofmt_xdate()  # Girar as datas para evitar sobreposição

    # Adicionar rótulos e título
    plt.legend()
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.grid(True)
    plt.title('Preços Reais vs Previsões (Gradient Boosting)')
    plt.show()

    """
    # 8
    st.subheader("Código:")
    st.code(codigo, language='python')

    # 9
    codigo = """predictions_next_week = predictions[-7:]  # Ajustar o número conforme necessário
    df_next_week_dates = df['Data'].iloc[-len(y_test):][-7:]  # Ajustar o número conforme necessário

    # Plotar os resultados
    plt.figure(figsize=(10, 5))
    # Certifique-se de reverter os dados para que as datas sejam plotadas em ordem cronológica
    plt.plot(df_next_week_dates[::-1], predictions_next_week[::-1], label='Previsão', color='orange', marker='o')

    # Formatar o eixo x para apresentar as datas
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()  # Auto formatar as datas para evitar sobreposição

    plt.title('Previsão dos Preços para a Próxima Semana')
    plt.xlabel('Data')
    plt.ylabel('Preço Previsto')
    plt.legend()
    plt.grid(True)
    plt.show()
    """
    # 9
    st.subheader("Código:")
    st.code(codigo, language='python')

    # 10
    codigo = """len(predictions)
    """
    # 10
    st.subheader("Código:")
    st.code(codigo, language='python')

    # 11
    codigo = """X_train
    """
    # 11
    st.subheader("Código:")
    st.code(codigo, language='python')

    # 12
    codigo = """
    # Carregar o DataFrame
    df = pd.read_excel('Petróleo.xlsx')
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values(by='Data', ascending=True).reset_index(drop=True)
    #df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
    #df['Preço'] = df['Preço'].astype(float)  # Certifique-se de que os preços são float

    # É uma boa prática criar recursos de atraso (lag features) para séries temporais
    # Vamos criar alguns para nosso modelo
    # Criar recursos de atraso (lag features)
    lags = 7
    for lag in range(1, lags + 1):
        df[f'Preço_lag_{lag}'] = df['Preço - petróleo bruto - Brent (FOB)'].shift(lag)

    # Removemos quaisquer linhas com valores NaN que foram criados ao fazer o shift
    df = df.dropna()

    # Preparando os dados para treinamento
    X = df[['Preço_lag_1']].values  # Inputs são os preços atrasados
    y = df['Preço - petróleo bruto - Brent (FOB)'].values  # Output é o preço atual

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    # Criar e treinar o modelo de Gradient Boosting
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, loss='squared_error')
    model.fit(X_train, y_train)

    # Fazer previsões
    predictions = model.predict(X_test)

    # Avaliar o modelo
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    # Fazer previsões para a próxima semana usando os últimos dados conhecidos
    last_known_data = X[-1].reshape(1, -1)
    next_week_predictions = []
    for _ in range(7):  # para cada dia da próxima semana
        next_day_pred = model.predict(last_known_data)[0]
        next_week_predictions.append(next_day_pred)
        last_known_data = np.roll(last_known_data, -1)
        last_known_data[0, -1] = next_day_pred

    # As datas correspondentes à próxima semana
    next_week_dates = pd.date_range(df['Data'].iloc[-1], periods=8)[1:]

    # Selecionar os dados da semana atual (últimos 7 dias do dataset)
    current_week_dates = df['Data'].iloc[-7:]
    current_week_prices = df['Preço - petróleo bruto - Brent (FOB)'].iloc[-7:]

    for week, pred in zip(next_week_dates, next_week_predictions):
        print(f'{week}: {pred:.2f}')

    # Plotar os preços reais da semana atual e as previsões para a próxima semana
    plt.figure(figsize=(10, 5))
    plt.plot(current_week_dates, current_week_prices, 'bo-', label='Preços Atuais')
    plt.plot(next_week_dates, next_week_predictions, 'r--o', label='Previsões para a Próxima Semana')

    # Formatar o eixo x para exibir datas
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()  # Ajustar formato das datas para evitar sobreposição

    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.title('Preços Reais e Previsões para as Últimas Duas Semanas')
    plt.legend()
    plt.grid(True)
    plt.show()

    """
    # 12
    st.subheader("Código:")
    st.code(codigo, language='python')

    # 13
    codigo = """df
    """
    # 13
    st.subheader("Código:")
    st.code(codigo, language='python')

    # Carrega os dados do arquivo Excel
    df = pd.read_excel("Petróleo.xlsx")

    # Mostra as primeiras linhas do DataFrame
    st.write(df.head())

    # Função para atualizar o DataFrame com novos dados
    def update_dataframe(df, new_data):
        # Converte a coluna 'Data' para datetime
        df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
        new_data['Data'] = pd.to_datetime(new_data['Data'], dayfirst=True)

        # Encontra a data mais recente no DataFrame existente
        last_date = df['Data'].max()

        # Filtra as novas linhas que são mais recentes do que a última data
        new_rows = new_data[new_data['Data'] > last_date]

        # Concatena os novos dados com o DataFrame existente se houver novas linhas
        if not new_rows.empty:
            updated_df = pd.concat([df, new_rows], ignore_index=True)
        else:
            updated_df = df

        return updated_df

    # Atualiza o DataFrame com novos dados do arquivo Excel
    path = 'ipea.csv'  # Especifique o nome do arquivo no mesmo diretório do seu script
    try:
        existing_df = pd.read_csv(path)
    except FileNotFoundError:
        existing_df = df  # Se o arquivo não existir, considere os dados atuais como o DataFrame existente

    # Atualiza o DataFrame existente com novos dados (carga incremental)
    updated_df = update_dataframe(existing_df, df)

    # Salva o DataFrame atualizado para o arquivo no mesmo diretório do script
    updated_df.to_csv(path, index=False)

    # Mostra as primeiras linhas do DataFrame atualizado
    st.write(updated_df.head())

    # Definir o dispositivo para a GPU se disponível
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Converter a coluna de data para datetime
    updated_df['Data'] = pd.to_datetime(updated_df['Data'], format='%Y-%m-%d')

    # Converter a coluna de data para um timestamp Unix
    updated_df['Timestamp'] = updated_df['Data'].apply(lambda x: x.timestamp())

    # Escalar a coluna de preços, já que os modelos de DL geralmente funcionam melhor com dados normalizados
    scaler = MinMaxScaler(feature_range=(-1, 1))
    updated_df['Preço - petróleo bruto - Brent (FOB)'] = scaler.fit_transform(updated_df['Preço - petróleo bruto - Brent (FOB)'].values.reshape(-1, 1)).astype('float32')

    # Preparar dados para o PyTorch
    X = updated_df['Timestamp'].values.astype('float32')  # A entrada do modelo será o timestamp
    y = updated_df['Preço - petróleo bruto - Brent (FOB)'].values.astype('float32')  # A saída do modelo serão os preços

    # Dividir o conjunto de dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Converter os dados para Tensor
    X_train_tensor = torch.tensor(X_train).view(-1, 1, 1)
    y_train_tensor = torch.tensor(y_train).view(-1, 1, 1)
    X_test_tensor = torch.tensor(X_test).view(-1, 1, 1)
    y_test_tensor = torch.tensor(y_test).view(-1, 1, 1)

    # Mover para o dispositivo apropriado
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    # Definir o modelo LSTM
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=200, output_size=1):
            super(LSTMModel, self).__init__()
            self.hidden_layer_size = hidden_layer_size

            self.lstm = nn.LSTM(input_size, hidden_layer_size ,num_layers=3)

            self.linear = nn.Linear(hidden_layer_size, output_size)

        def forward(self, input_seq):
            lstm_out, _ = self.lstm(input_seq)
            predictions = self.linear(lstm_out.view(len(input_seq), -1))
            return predictions[-1]

    # Instanciar o modelo
    model = LSTMModel().to(device)

    # Definir a função de perda e o otimizador
    loss_function = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.0001)


    st.write("Treinamento do modelo iniciado. Aguarde...")
    # Treinar o modelo
    epochs = 10
    print_loss = True  # Variável para controlar a impressão das perdas no Epoch 0
    for i in range(epochs):
        for seq, labels in zip(X_train_tensor, y_train_tensor):
            optimizer.zero_grad()

            model.hidden = (torch.zeros(3, 1, model.hidden_layer_size).to(device),
                            torch.zeros(3, 1, model.hidden_layer_size).to(device))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

            if i == 0 and print_loss:  # Imprime apenas para o Epoch 0 se print_loss for True
                st.write(f'Epoch {i} loss: {single_loss.item()}')
                print_loss = False  # Impede a impressão de mais perdas

            if i == 0 and single_loss.item() < 0.0774:  # Se a perda for menor que 0.0774, pare a impressão
                break

    

    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(len(X_test)):
            seq = X_test_tensor[i : i + 1]
            preds.append(model(seq).cpu().numpy()[0])

    # Inverter a escala dos preços para a escala original
    actual_predictions = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

    # Mostrar o gráfico dos preços reais vs previsões
    plt.figure(figsize=(12, 6))
    plt.plot(updated_df['Data'][-len(y_test):], y_test, label="Actual Prices", marker='o')
    plt.plot(updated_df['Data'][-len(y_test):], actual_predictions, label="Predicted Prices", marker='o')
    plt.xlabel('Data')
    plt.ylabel('Preço - petróleo bruto - Brent (FOB)')
    plt.title('Preços Reais vs Previsões')
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Mostrar o gráfico no Streamlit
    st.pyplot(plt)

    # Carregar o DataFrame
    df = pd.read_excel('Petróleo.xlsx')
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values(by='Data', ascending=True).reset_index(drop=True)

    # Criar recursos de atraso (lag features) para séries temporais
    for lag in range(1, 2):  # Criar atrasos de 1 dia até 3 dias
        df[f'Preço_lag_{lag}'] = df['Preço - petróleo bruto - Brent (FOB)'].shift(lag)

   # Remover linhas com valores NaN que foram criados ao fazer o shift
    df = df.dropna()

    # Preparar os dados para treinamento
    X = df[['Preço_lag_1']].values  # Inputs são os preços atrasados
    y = df['Preço - petróleo bruto - Brent (FOB)'].values  # Output é o preço atual

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)

    # Criar e treinar o modelo de Gradient Boosting
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, loss='squared_error')
    model.fit(X_train, y_train)

    # Fazer previsões
    predictions = model.predict(X_test)

    # Avaliar o modelo
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Plotar resultados reais vs previstos
    def plot_predictions(actual, predictions, dates):
        plt.figure(figsize=(15, 5))
        plt.plot(dates, actual, label='Real')
        plt.plot(dates, predictions, label='Previsão')

        # Melhorar a formatação do eixo x
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Formatar datas como 'Ano-Mês-Dia'
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())             # Escolher automaticamente a localização das datas
        plt.gcf().autofmt_xdate()  # Girar as datas para evitar sobreposição

        # Adicionar rótulos e título
        plt.legend()
        plt.xlabel('Data')
        plt.ylabel('Preço')
        plt.grid(True)
        plt.title('Preços Reais vs Previsões (Gradient Boosting)')

    # Função principal do Streamlit
    def main():
        st.title('Previsão de Preços de Petróleo com Gradient Boosting')
        st.subheader('Avaliação do Modelo')
        st.write("Mean Squared Error:", mse)
        st.write("Mean Absolute Error:", mae)

        st.subheader('Gráfico de Preços Reais vs Previsões')
        plot_predictions(y_test, predictions, df['Data'].iloc[-len(predictions):])
        st.pyplot()

        predictions_next_week = predictions[-7:]  # Ajustar o número conforme necessário
        df_next_week_dates = df['Data'].iloc[-len(y_test):][-7:]  # Ajustar o número conforme necessário

        # Plotar os resultados
        plt.figure(figsize=(10, 5))
        # Certifique-se de reverter os dados para que as datas sejam plotadas em ordem cronológica
        plt.plot(df_next_week_dates[::-1], predictions_next_week[::-1], label='Previsão', color='orange', marker='o')

        # Formatar o eixo x para apresentar as datas
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gcf().autofmt_xdate()  # Auto formatar as datas para evitar sobreposição

        plt.title('Previsão dos Preços para a Próxima Semana')
        plt.xlabel('Data')
        plt.ylabel('Preço Previsto')
        plt.legend()
        plt.grid(True)
        plt.show()

        print("Tamanho das previsões:", len(predictions))
        print("Conjunto de treinamento:")
        print(X_train)

        # Carregar o DataFrame
        df = pd.read_excel('Petróleo.xlsx')
        df['Data'] = pd.to_datetime(df['Data'])
        df = df.sort_values(by='Data', ascending=True).reset_index(drop=True)

        # Carregar o DataFrame
    df = pd.read_excel('Petróleo.xlsx')
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values(by='Data', ascending=True).reset_index(drop=True)

    # Criar recursos de atraso (lag features) para séries temporais
    for lag in range(1, 2):  # Criar atrasos de 1 dia até 3 dias
        df[f'Preço_lag_{lag}'] = df['Preço - petróleo bruto - Brent (FOB)'].shift(lag)

    # Remover linhas com valores NaN que foram criados ao fazer o shift
    df = df.dropna()

    # Preparar os dados para treinamento
    X = df[['Preço_lag_1']].values  # Inputs são os preços atrasados
    y = df['Preço - petróleo bruto - Brent (FOB)'].values  # Output é o preço atual

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)

    # Criar e treinar o modelo de Gradient Boosting
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, loss='squared_error')
    model.fit(X_train, y_train)

    # Fazer previsões
    predictions = model.predict(X_test)

    # Avaliar o modelo
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Função para plotar os resultados
    def plot_predictions(actual, predictions, dates):
        plt.figure(figsize=(15, 5))
        plt.plot(dates, actual, label='Real')
        plt.plot(dates, predictions, label='Previsão')

        # Melhorar a formatação do eixo x
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Formatar datas como 'Ano-Mês-Dia'
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())             # Escolher automaticamente a localização das datas
        plt.gcf().autofmt_xdate()  # Girar as datas para evitar sobreposição

        # Adicionar rótulos e título
        plt.legend()
        plt.xlabel('Data')
        plt.ylabel('Preço')
        plt.grid(True)
        plt.title('Preços Reais vs Previsões (Gradient Boosting)')

    # Aplicar a função para plotar
    plot_predictions(y_test, predictions, df['Data'].iloc[-len(predictions):])

    # Exibir o gráfico no Streamlit
    st.pyplot()

    predictions_next_week = predictions[-7:]  # Ajustar o número conforme necessário
    df_next_week_dates = df['Data'].iloc[-len(y_test):][-7:]  # Ajustar o número conforme necessário

    # Plotar os resultados
    plt.figure(figsize=(10, 5))
    # Certifique-se de reverter os dados para que as datas sejam plotadas em ordem cronológica
    plt.plot(df_next_week_dates[::-1], predictions_next_week[::-1], label='Previsão', color='orange', marker='o')

    # Formatar o eixo x para apresentar as datas
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()  # Auto formatar as datas para evitar sobreposição

    plt.title('Previsão dos Preços para a Próxima Semana')
    plt.xlabel('Data')
    plt.ylabel('Preço Previsto')
    plt.legend()
    plt.grid(True)
    plt.show()

    st.write("Comprimento das previsões:", len(predictions))

    # Carregar o DataFrame
    df = pd.read_excel('Petróleo.xlsx')
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values(by='Data', ascending=True).reset_index(drop=True)

    # É uma boa prática criar recursos de atraso (lag features) para séries temporais
    # Vamos criar alguns para nosso modelo
    # Criar recursos de atraso (lag features)
    lags = 7
    for lag in range(1, lags + 1):
        df[f'Preço_lag_{lag}'] = df['Preço - petróleo bruto - Brent (FOB)'].shift(lag)

    # Remover linhas com valores NaN que foram criados ao fazer o shift
    df = df.dropna()

    # Preparar os dados para treinamento
    X = df[['Preço_lag_1']].values  # Inputs são os preços atrasados
    y = df['Preço - petróleo bruto - Brent (FOB)'].values  # Output é o preço atual

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    # Criar e treinar o modelo de Gradient Boosting
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, loss='squared_error')
    model.fit(X_train, y_train)

    # Fazer previsões
    predictions = model.predict(X_test)

    # Avaliar o modelo
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Exibir métricas de avaliação
    st.write("Mean Squared Error:", mse)
    st.write("Mean Absolute Error:", mae)

    # Fazer previsões para a próxima semana usando os últimos dados conhecidos
    last_known_data = X[-1].reshape(1, -1)
    next_week_predictions = []
    for _ in range(7):  # para cada dia da próxima semana
        next_day_pred = model.predict(last_known_data)[0]
        next_week_predictions.append(next_day_pred)
        last_known_data = np.roll(last_known_data, -1)
        last_known_data[0, -1] = next_day_pred

    # As datas correspondentes à próxima semana
    next_week_dates = pd.date_range(df['Data'].iloc[-1], periods=8)[1:]

    # Selecionar os dados da semana atual (últimos 7 dias do dataset)
    current_week_dates = df['Data'].iloc[-7:]
    current_week_prices = df['Preço - petróleo bruto - Brent (FOB)'].iloc[-7:]

    # Imprimir previsões para a próxima semana
    st.write("Previsões para a Próxima Semana:")
    for week, pred in zip(next_week_dates, next_week_predictions):
        st.write(f'{week}: {pred:.2f}')

    # Plotar os preços reais da semana atual e as previsões para a próxima semana
    plt.figure(figsize=(10, 5))
    plt.plot(current_week_dates, current_week_prices, 'bo-', label='Preços Atuais')
    plt.plot(next_week_dates, next_week_predictions, 'r--o', label='Previsões para a Próxima Semana')

    # Formatar o eixo x para exibir datas
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()  # Ajustar formato das datas para evitar sobreposição

    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.title('Preços Reais e Previsões para as Últimas Duas Semanas')
    plt.legend()
    plt.grid(True)

    # Exibir o gráfico no Streamlit
    st.pyplot()

    # Texto explicativo sobre o modelo LSTM
    st.write("O modelo utilizado neste código é uma Rede Neural Recorrente (RNN) com uma camada Long Short-Term Memory (LSTM). LSTM é um tipo especial de RNN que é capaz de aprender e lembrar padrões de longo prazo em sequências de dados.")
    st.write("O resultado desse modelo LSTM representa as previsões para os valores futuros da série temporal de interesse. Em outras palavras, o modelo tenta prever os próximos valores da série temporal com base nos padrões aprendidos nos dados de treinamento.")
    st.write("Mais especificamente, o resultado são as previsões para o próximo ponto na série temporal após os dados de treinamento. Essas previsões são baseadas nas informações históricas fornecidas como entrada para o modelo durante o treinamento.")

    # Resultados do modelo LSTM
    st.write("Resultados do modelo:")
    st.write("- MSE:", 2.836992512397856)
    st.write("- MAE:", 1.1848350437487067)

    # Análise do desempenho do modelo
    st.write("O modelo parece ter um desempenho razoável, pois tanto o MSE quanto o MAE não são extremamente altos. No entanto, é difícil tirar conclusões definitivas sem um contexto mais detalhado sobre os dados e as expectativas em relação ao modelo.")
    st.write("É recomendado também comparar esses valores com os de outros modelos ou com métricas de linha de base para avaliar melhor o desempenho relativo do modelo.")

def apresentacao_Modelo_Machine_Learning_ARIMA():
    st.header("Modelo Machine Learning - ARIMA")

    codigo = """
    # Carregando os dados do petróleo
    df = pd.read_excel('Petróleo.xlsx')

    # Converter a coluna 'Data' para datetime
    df['Data'] = pd.to_datetime(df['Data'])

    # Definir a coluna 'Data' como índice, para análise de série temporais
    df.set_index('Data', inplace=True)

    # Dividir os dados em conjunto de treinamento e teste (80% treinamento, 20% teste)
    train_size = int(len(df) * 0.8)
    train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

    # Criar e treinar o modelo SARIMA
    # Objeto SARIMAX criado e treinado usando o conjunto de treinamento.
    # O modelo SARIMA foi configurado com ordens (1, 1, 1) para os componentes ARIMA e ordens sazonais (1, 1, 1, 12) para capturar sazonalidade anual.
    model = SARIMAX(train_data['Preço - petróleo bruto - Brent (FOB)'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()

    # Fazer previsões sobre os preços do petróleo para o conjunto de teste.
    predictions = model_fit.forecast(steps=len(test_data))

    # Calcular métricas de avaliação para que sejam calculadas e comparadas com os valores reais das previsões no conjunto de teste.
    mse = mean_squared_error(test_data, predictions)
    mae = mean_absolute_error(test_data, predictions)

    # Plotar previsões
    st.subheader('Previsões do Modelo SARIMA para Preço do Petróleo Brent')
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data, label='Dados de Treinamento')
    plt.plot(test_data.index, test_data, label='Dados de Teste')
    plt.plot(test_data.index, predictions, label='Previsões', color='red')
    plt.title('Previsões do Modelo SARIMA para Preço do Petróleo Brent')
    plt.xlabel('Data')
    plt.ylabel('Preço do Petróleo (USD)')
    plt.legend()
    plt.grid(True)
    st.pyplot()

    # Exibir métricas de avaliação
    st.write(f'MSE (Mean Squared Error): [mse]')
    st.write(f'MAE (Mean Absolute Error): {mae]')

    st.write("Treinamento do modelo iniciado. Aguarde...")

    # Carregar os dados do petróleo
    df = pd.read_excel('Petróleo.xlsx')

    # Converter a coluna 'Data' para datetime
    df['Data'] = pd.to_datetime(df['Data'])

    # Definir a coluna 'Data' como índice
    df.set_index('Data', inplace=True)

    # Dividir os dados em conjunto de treinamento e teste (80% treinamento, 20% teste)
    train_size = int(len(df) * 0.8)
    train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

    # Definir os hiperparâmetros a serem testados
    p = range(0, 2)  # Ordem de autorregressão
    d = range(0, 2)  # Ordem de diferenciação
    q = range(0, 2)  # Ordem da média móvel
    P = range(0, 2)  # Ordem sazonal de autorregressão
    D = range(0, 2)  # Ordem sazonal de diferenciação
    Q = range(0, 2)  # Ordem sazonal da média móvel
    s = 12  # Frequência sazonal (12 para dados mensais)

    # Criar todas as combinações possíveis de hiperparâmetros
    hyperparameters = list(product(p, d, q, P, D, Q))

    # Limitar o número de combinações de hiperparâmetros
    hyperparameters = hyperparameters[:20]

    # Inicializar listas para armazenar resultados
    best_mse = np.inf
    best_params = None

    # Listas para armazenar resultados para plotagem
    mse_results = []
    params_results = []

    # Loop sobre todas as combinações de hiperparâmetros
    for param in hyperparameters:
        try:
            # Ajustar o modelo SARIMA
            model = sm.tsa.statespace.SARIMAX(train_data['Preço - petróleo bruto - Brent (FOB)'],
                                            order=(param[0], param[1], param[2]),
                                            seasonal_order=(param[3], param[4], param[5], s),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            model_fit = model.fit(disp=False)

            # Fazer previsões
            predictions = model_fit.forecast(steps=len(test_data))

            # Calcular MSE
            mse = mean_squared_error(test_data, predictions)

            # Atualizar melhores parâmetros se necessário
            if mse < best_mse:
                best_mse = mse
                best_params = param

            # Armazenar resultados
            mse_results.append(mse)
            params_results.append(param)

        except:
            continue

    # Plotar o MSE em relação aos diferentes conjuntos de hiperparâmetros
    st.pyplot(plt.figure(figsize=(12, 6)))
    plt.plot(range(len(mse_results)), mse_results)
    plt.xlabel('Conjunto de Hiperparâmetros')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE para Diferentes Conjuntos de Hiperparâmetros')
    st.pyplot()

    # Imprimir os melhores hiperparâmetros e MSE correspondente
    st.write(f"Melhores hiperparâmetros encontrados: [best_params}")
    st.write(f"MSE correspondente: [best_mse}")
    """
    # 1
    st.subheader("Código:")
    st.code(codigo, language='python')

    # Carregando os dados do petróleo
    df = pd.read_excel('Petróleo.xlsx')

    # Converter a coluna 'Data' para datetime
    df['Data'] = pd.to_datetime(df['Data'])

    # Definir a coluna 'Data' como índice, para análise de série temporais
    df.set_index('Data', inplace=True)

    # Dividir os dados em conjunto de treinamento e teste (80% treinamento, 20% teste)
    train_size = int(len(df) * 0.8)
    train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

    # Criar e treinar o modelo SARIMA
    # Objeto SARIMAX criado e treinado usando o conjunto de treinamento.
    # O modelo SARIMA foi configurado com ordens (1, 1, 1) para os componentes ARIMA e ordens sazonais (1, 1, 1, 12) para capturar sazonalidade anual.
    model = SARIMAX(train_data['Preço - petróleo bruto - Brent (FOB)'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()

    # Fazer previsões sobre os preços do petróleo para o conjunto de teste.
    predictions = model_fit.forecast(steps=len(test_data))

    # Calcular métricas de avaliação para que sejam calculadas e comparadas com os valores reais das previsões no conjunto de teste.
    mse = mean_squared_error(test_data, predictions)
    mae = mean_absolute_error(test_data, predictions)

    # Plotar previsões
    st.subheader('Previsões do Modelo SARIMA para Preço do Petróleo Brent')
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data, label='Dados de Treinamento')
    plt.plot(test_data.index, test_data, label='Dados de Teste')
    plt.plot(test_data.index, predictions, label='Previsões', color='red')
    plt.title('Previsões do Modelo SARIMA para Preço do Petróleo Brent')
    plt.xlabel('Data')
    plt.ylabel('Preço do Petróleo (USD)')
    plt.legend()
    plt.grid(True)
    st.pyplot()

    # Exibir métricas de avaliação
    st.write(f'MSE (Mean Squared Error): {mse}')
    st.write(f'MAE (Mean Absolute Error): {mae}')

    # Texto para exibição no Streamlit
    st.write('**Avaliação do Modelo SARIMA:**')
    st.write('O MSE (Erro Quadrático Médio) é aproximadamente 595.13.')
    st.write('O MAE (Erro Médio Absoluto) é aproximadamente 19.31.')
    st.write('Esses valores indicam que, em média, as previsões do modelo estão desviando do valor real dos preços do petróleo em torno de $19.31 (para o MAE) e que os erros quadráticos médios das previsões estão em torno de 595.13 (para o MSE).')
    st.write('Em resumo, os valores relativamente altos do MSE e do MAE sugerem que o modelo SARIMA pode não estar fornecendo previsões muito precisas para os preços do petróleo Brent.')
    st.write('Considerando esses resultados, podemos concluir que o modelo tem um desempenho razoável na previsão do preço do petróleo Brent. No entanto, é possível que haja espaço para melhorias, especialmente se os valores do MSE e do MAE puderem ser reduzidos por meio de ajustes no modelo ou na seleção de diferentes algoritmos de previsão.')
    
    st.write("Treinamento do modelo iniciado. Aguarde...")

    # Carregar os dados do petróleo
    df = pd.read_excel('Petróleo.xlsx')

    # Converter a coluna 'Data' para datetime
    df['Data'] = pd.to_datetime(df['Data'])

    # Definir a coluna 'Data' como índice
    df.set_index('Data', inplace=True)

    # Dividir os dados em conjunto de treinamento e teste (80% treinamento, 20% teste)
    train_size = int(len(df) * 0.8)
    train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

    # Definir os hiperparâmetros a serem testados
    p = range(0, 2)  # Ordem de autorregressão
    d = range(0, 2)  # Ordem de diferenciação
    q = range(0, 2)  # Ordem da média móvel
    P = range(0, 2)  # Ordem sazonal de autorregressão
    D = range(0, 2)  # Ordem sazonal de diferenciação
    Q = range(0, 2)  # Ordem sazonal da média móvel
    s = 12  # Frequência sazonal (12 para dados mensais)

    # Criar todas as combinações possíveis de hiperparâmetros
    hyperparameters = list(product(p, d, q, P, D, Q))

    # Limitar o número de combinações de hiperparâmetros
    hyperparameters = hyperparameters[:20]

    # Inicializar listas para armazenar resultados
    best_mse = np.inf
    best_params = None

    # Listas para armazenar resultados para plotagem
    mse_results = []
    params_results = []

    # Loop sobre todas as combinações de hiperparâmetros
    for param in hyperparameters:
        try:
            # Ajustar o modelo SARIMA
            model = sm.tsa.statespace.SARIMAX(train_data['Preço - petróleo bruto - Brent (FOB)'],
                                            order=(param[0], param[1], param[2]),
                                            seasonal_order=(param[3], param[4], param[5], s),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            model_fit = model.fit(disp=False)

            # Fazer previsões
            predictions = model_fit.forecast(steps=len(test_data))

            # Calcular MSE
            mse = mean_squared_error(test_data, predictions)

            # Atualizar melhores parâmetros se necessário
            if mse < best_mse:
                best_mse = mse
                best_params = param

            # Armazenar resultados
            mse_results.append(mse)
            params_results.append(param)

        except:
            continue

    # Plotar o MSE em relação aos diferentes conjuntos de hiperparâmetros
    st.pyplot(plt.figure(figsize=(12, 6)))
    plt.plot(range(len(mse_results)), mse_results)
    plt.xlabel('Conjunto de Hiperparâmetros')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE para Diferentes Conjuntos de Hiperparâmetros')
    st.pyplot()

    # Imprimir os melhores hiperparâmetros e MSE correspondente
    st.write(f"Melhores hiperparâmetros encontrados: {best_params}")
    st.write(f"MSE correspondente: {best_mse}")

    # Texto explicativo sobre o código
    st.write("Neste código, estamos ajustando modelos SARIMA para todas as combinações possíveis dos hiperparâmetros especificados (p, d, q, P, D, Q) usando Grid Search. O modelo que produz o menor MSE nos dados de teste é considerado como o melhor modelo.")
    st.write("Este código realiza uma busca de hiperparâmetros para encontrar os melhores parâmetros para um modelo SARIMA (Seasonal Autoregressive Integrated Moving Average) ajustado aos dados de preços do petróleo, afim de melhorar o resultado final do nosso modelo.")
    st.write("O objetivo desse código é encontrar os melhores hiperparâmetros para o modelo SARIMA que proporcionem as melhores previsões para os dados de preços do petróleo.")
    st.write("Isso significa que, após testar diferentes combinações de hiperparâmetros, o modelo SARIMA obteve seu melhor desempenho com os seguintes parâmetros:")
    st.write("- Ordem de autorregressão (p): 0")
    st.write("- Ordem de diferenciação (d): 1")
    st.write("- Ordem da média móvel (q): 0")
    st.write("- Ordem sazonal de autorregressão (P): 0")
    st.write("- Ordem sazonal de diferenciação (D): 1")
    st.write("- Ordem sazonal da média móvel (Q): 1")
    st.write("O correspondente Mean Squared Error (MSE), que é uma medida do quão próximas as previsões do modelo estão dos valores reais, foi de 621.27927059005.")
    st.write("Em resumo, esses são os melhores parâmetros encontrados para o modelo SARIMA que foram capazes de minimizar o erro quadrático médio em relação aos dados de teste.")
 

if __name__ == "__main__":
    main()
