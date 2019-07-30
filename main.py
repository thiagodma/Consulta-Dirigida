import time
import consulta_dirigida_funcoes as cdf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.cluster import hierarchy

#stopwords definidas
stop_words = cdf.define_stop_words()

#Lê a planilha 
df = pd.read_excel('Consulta dirigida de Laboratórios Clínicos.xlsx',dtype=str).fillna('nao teve resposta')

resposta_id = list(df['ID da resposta'])

it=15 #indice da coluna que contém a resposta
it_aux = 1 #será usado apenas para fazer referência às perguntas como primeira, segunda, terceira, etc
analise = []
while it <= 63:
    
    #pega o primeiro conjunto de perguntas
    respostas = list(df.iloc[:,it])
    resposta_id = list(df['ID da resposta'])
    
    #Retira as respostas vazias
    respostas_aux = []
    respostas_tratadas_aux = []
    resposta_id_aux = []
    for i in range(len(respostas)):
        if respostas[i] != 'nao teve resposta':
            respostas_aux.append(respostas[i])
            respostas_tratadas_aux.append(cdf.trata_respostas(respostas[i],stop_words))
            resposta_id_aux.append(resposta_id[i])
            
    respostas = respostas_aux
    respostas_tratadas = respostas_tratadas_aux
    resposta_id = resposta_id_aux
    
    #Faz o stemming
    textos_stem = cdf.stem(respostas_tratadas)
    idxs = [i for i,value in enumerate(textos_stem) if value=='']
    for idx in idxs: textos_stem[idx] = 'fora de padrao'
    
    #Vetorizando e aplicando o tfidf
    vec = CountVectorizer()
    bag_palavras = vec.fit_transform(textos_stem)
    feature_names = vec.get_feature_names()
    base_tfidf = TfidfTransformer().fit_transform(bag_palavras)
    base_tfidf = base_tfidf.todense()
    
    #Reduzindo a dimensionalidade
    base_tfidf_reduced = cdf.SVD(60, base_tfidf)
    
    #Clustering
    print('Começou a clusterização.')
    t = time.time()
    clusters_por_cosseno = hierarchy.linkage(base_tfidf_reduced,"average", metric="cosine") #pode testar metric="euclidean" também
    #plt.figure()
    #dn = hierarchy.dendrogram(clusters_por_cosseno)
    elpsd = time.time() - t
    print('Tempo para fazer a clusterização: ' + str(elpsd) + '\n')
    
    # Separa a que Cluster pertence cada texto, pela ordem na lista de textos,
    # dado o parâmetro de limite de dissimilaridade threshold
    limite_dissimilaridade = 0.9
    id_clusters = hierarchy.fcluster(clusters_por_cosseno, limite_dissimilaridade, criterion="distance")
    
    #Tentando visualizar os dados e vendo o número de amostras por cluster
    analise.append(cdf.analisa_clusters(base_tfidf, id_clusters))
    
    
    print('Foram encontradas ' + str(max(id_clusters)) + ' clusters\n')
    
    #Exporta as tabelas
    cdf.generate_csvs_for_powerbi(analise[it_aux-1],it_aux, id_clusters, resposta_id, respostas)
    it = it+2
    it_aux = it_aux+1