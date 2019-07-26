import time
import consulta_dirigida_funcoes as cdf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.cluster import hierarchy

stop_words = cdf.define_stop_words()

df = pd.read_excel('Consulta dirigida de Laboratórios Clínicos.xlsx',dtype=str).fillna('nao teve resposta')

resposta_id = list(df['ID da resposta'])

#pega o primeiro conjunto de perguntas
respostas = list(df.iloc[:,11])



respostas_tratadas = [cdf.trata_respostas(resposta,stop_words) for resposta in respostas]

#Faz o stemming
textos_stem = cdf.stem(respostas_tratadas)
idxs = [i for i,value in enumerate(textos_stem) if value=='']
for idx in idxs: textos_stem[idx] = 'nao tev respost'

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
plt.figure()
dn = hierarchy.dendrogram(clusters_por_cosseno)
elpsd = time.time() - t
print('Tempo para fazer a clusterização: ' + str(elpsd) + '\n')

# Separa a que Cluster pertence cada texto, pela ordem na lista de textos,
# dado o parâmetro de limite de dissimilaridade threshold
limite_dissimilaridade = 0.9
id_clusters = hierarchy.fcluster(clusters_por_cosseno, limite_dissimilaridade, criterion="distance")

#Tentando visualizar os dados e vendo o número de amostras por cluster
analise = cdf.analisa_clusters(base_tfidf, id_clusters)

#Colocando em dataframes
X = pd.DataFrame(id_clusters,columns=['cluster_id'])
Y = pd.DataFrame(resposta_id ,columns=['resposta_id'])
Z = X.join(Y)

print('Foram encontradas ' + str(max(Z['cluster_id'])) + ' clusters\n')

#Exporta as tabelas
cdf.generate_csvs_for_powerbi(analise,Z,respostas,respostas_tratadas)