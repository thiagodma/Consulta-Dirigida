import re, unicodedata, time, nltk
from stop_words import get_stop_words
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


#Pega stop_words e as trata para ficarem no formato correto
def define_stop_words():
    
    stop_words = get_stop_words('portuguese')
    stop_words = stop_words + nltk.corpus.stopwords.words('portuguese')
    stop_words = stop_words + ['art','dou','secao','pag','pagina', 'in', 'inc', 'obs', 'sob', 'ltda']
    stop_words = stop_words + ['ndash', 'mdash', 'lsquo','rsquo','ldquo','rdquo','bull','hellip','prime','lsaquo','rsaquo','frasl', 'ordm']
    stop_words = stop_words + ['prezado', 'prezados', 'prezada', 'prezadas', 'gereg', 'ggali','usuario', 'usuaria', 'deseja','gostaria', 'boa tarde', 'bom dia', 'boa noite']
    stop_words = list(dict.fromkeys(stop_words))
    stop_words = ' '.join(stop_words)
    #As stop_words vem com acentos/cedilhas. Aqui eu tiro os caracteres indesejados
    stop_words = limpa_utf8(stop_words)
    return stop_words


def trata_respostas(texto, stop_words):
    
    if texto=='nao teve resposta': return texto
    
    #converte todos caracteres para letra minúscula
    texto_lower = texto.lower()
    texto_lower = re.sub(' +', ' ', texto_lower)
    
    #tira sites
    texto_sem_sites =  re.sub('(http|www)[^ ]+','',texto_lower)
    
    #Remove acentos e pontuação
    texto_sem_acento_pontuacao = limpa_utf8(texto_sem_sites)
    
    #Retira numeros romanos e stopwords
    texto_sem_acento_pontuacao = texto_sem_acento_pontuacao.split()
    texto_sem_stopwords = [roman2num(palavra,stop_words) for palavra in texto_sem_acento_pontuacao]
    texto_sem_stopwords = ' '.join(texto_sem_stopwords)
    
    #Remove hifens e barras
    texto_sem_hifens_e_barras = re.sub('[-\/]', ' ', texto_sem_stopwords)
    
    #Troca qualquer tipo de espacamento por espaço
    texto_sem_espacamentos = re.sub(r'\s', ' ', texto_sem_hifens_e_barras)
    
    #Remove pontuacao e digitos
    texto_limpo = re.sub('[^A-Za-z]', ' ' , texto_sem_espacamentos)
    
    
    #Remove espaços extras
    texto_limpo = re.sub(' +', ' ', texto_limpo)    
    
    return texto_limpo
    
    
# Recodificacao em utf8, removendo cedilhas acentos e coisas de latin
def limpa_utf8(texto):    

    texto = texto.split()
    texto_tratado = []
    for palavra in texto:
        # Unicode normalize transforma um caracter em seu equivalente em latin.
        nfkd = unicodedata.normalize('NFKD', palavra)
        palavra_sem_acento = u"".join([c for c in nfkd if not unicodedata.combining(c)])
        texto_tratado.append(palavra_sem_acento)
            
    return ' '.join(texto_tratado)


def roman2num(roman, stop_words, values={'m': 1000, 'd': 500, 'c': 100, 'l': 50,
                                'x': 10, 'v': 5, 'i': 1}):
    roman = limpa_utf8(roman)
    
    #remove stopwords
    if roman in stop_words: 
        return ''
      

    #como eu vou tirar numeros de qualquer forma, posso simplesmente retornar um numero
    if(len(roman) < 2 ):
        return str(1)

    if (roman == ''): return ''
    out = re.sub('[^mdclxvi]', '', roman)
    if (len(out) != len(roman)):
        return roman

    numbers = []
    for char in roman:
        numbers.append(values[char])
    total = 0
    if(len(numbers) > 1):
        for num1, num2 in zip(numbers, numbers[1:]):
            if num1 >= num2:
                total += num1
            else:
                total -= num1
        return str(total + num2)
    else:
        return str(numbers[0])

#Reduz a dimensionalidade dos dados
def SVD(dim,base_tfidf):
    print('Começou a redução de dimensionalidade.')
    t = time.time()
    svd = TruncatedSVD(n_components = dim, random_state = 42)
    base_tfidf_reduced = svd.fit_transform(base_tfidf)
    print('Número de dimensoes de entrada: ' + str(base_tfidf.shape[1]))
    print(str(dim) + ' dimensões explicam ' + str(svd.explained_variance_ratio_.sum()) + ' da variância.')
    elpsd = time.time() - t
    print('Tempo para fazer a redução de dimensionalidade: ' + str(elpsd) + '\n')
    return base_tfidf_reduced


#Visualiza as cluster definidas pelo algoritmo. Além disso também retorna o número
#de normas por cluster.
def analisa_clusters(base_tfidf, id_clusters):
    
    clusters = np.unique(id_clusters)
    
    #inicializa o output da funcao
    n_normas = np.zeros(len(clusters)) #numero de normas pertencentes a uma cluster
    
    #reduz a dimensionalidade para 2 dimensoes
    base_tfidf_reduced = SVD(2,base_tfidf)
    X = base_tfidf_reduced[:,0]
    Y = base_tfidf_reduced[:,1]
    
    colors = cm.rainbow(np.linspace(0, 1, len(n_normas)))

    for cluster, color in zip(clusters, colors):
        idxs = np.where(id_clusters == cluster) #a primeira cluster não é a 0 e sim a 1
        n_normas[cluster-1] = len(idxs[0])
        x = X[idxs[0]]
        y = Y[idxs[0]]
        plt.scatter(x, y, color=color)
    
    return n_normas

#Faz os stemming nas palavras utilizando o pacote NLTK com o RSLP Portuguese stemmer
def stem(resolucoes):
    
    print('Comecou a fazer o stemming.')
    t = time.time()
    #Inicializo a lista que será o retorno da funcao
    res = []
    
    #inicializando o objeto stemmer
    stemmer = nltk.stem.RSLPStemmer()
    
    for resolucao in resolucoes:
        #Faz o stemming para cada palavra na resolucao
        palavras_stemmed_resolucao = [stemmer.stem(word) for word in resolucao.split()]
        #Faz o append da resolucao que passou pelo stemming
        res.append(" ".join(palavras_stemmed_resolucao))
    
    print('Tempo para fazer o stemming: ' + str(time.time() - t) + '\n')
        
    return res

def mostra_conteudo_clusters(cluster,n_amostras,respostas,it_aux):
    df = pd.read_csv(str(it_aux) + 'texto_respostas_por_cluster.csv', sep='|')
    a = df[df['cluster_id'] == cluster]
    
    if a.shape[0] >= n_amostras: mostra = a.sample(n_amostras,random_state = 42)
    else : mostra = a
    
    fo = open(r'conteudo_cluster'+str(cluster)+'_n_'+str(a.shape[0])+'.txt', 'w+')
    
    for i in range(mostra.shape[0]):
        id_resposta = mostra.iloc[i,1]
        fo.writelines('ID da Resposta: ' + str(id_resposta) + '\n')
        idx = mostra.index[mostra['resposta_id'] == id_resposta].tolist()[0]
        fo.writelines(respostas[idx])
        fo.write('\n\n')
            
    fo.close()
    
'''
def generate_csvs_for_powerbi(analise, Z, perguntas_id, respostas, respostas_tratadas,it_aux):
    
    clusters = [i for i in range(1,len(analise)+1)]
    
    #Prepara a tabela que indica o número de perguntas por cluster
    d={'pergunta_id':perguntas_id,'cluster_id':clusters,'numero_de_respostas':analise}
    df = pd.DataFrame(d)
    #exporta a tabela para um csv
    df.to_csv('info_cluster.csv',sep='|',index=False,encoding='utf-8')
    
    '''
    #adiciona as keywords de cada cluster no csv
    palavras_relevantes = [mostra_palavras_relevantes(cluster,respostas_tratadas,10) for cluster in clusters]
    df['Keywords'] = 'default'
    for i in range(len(clusters)):
        df.iloc[i,2] = palavras_relevantes[i]
    df.to_csv('info_cluster.csv',sep='|',index=False,encoding='utf-8')
    '''
    
    #Prepara a tabela que tem a pergunta, o indentificador da pergunta
    #e a qual cluster a pergunta pertence
    Z['pergunta_sem_processamento'] = 'default'
    Z['pergunta_com_processamento'] = 'default'
    for i in range(Z.shape[0]):
        id_pergunta = Z.iloc[i,1]
        idx = Z.index[Z['resposta_id'] == id_pergunta].tolist()[0]
        #Adiciona a pergunta sem processamento na coluna correspondente
        Z.iloc[i,2] = respostas[idx]
        #Adiciona a pergunta com processamento na coluna correspondente
        Z.iloc[i,3] = respostas_tratadas[idx]
        
    Z.to_csv('texto_respostas_por_cluster.csv',sep='|',index=False,encoding='utf-8')
'''

def generate_csvs_for_powerbi(analise, Z):
    
    
    
    with open('info_cluster', 'a') as f:
        df.to_csv(f, header=False)
    
    
    
    
def generate_wordcloud(cluster,it_aux,stop_words):
    
    #importa o csv que tem a informação das clusters
    df = pd.read_csv('texto_respostas_por_cluster.csv', sep='|')
    a = df[df['cluster_id'] == cluster]
    
    L = list(a.iloc[:,3])
    text = '\n'.join(L)
    
    
    wordcloud = WordCloud(stopwords=stop_words.split()+['laboratorio','laboratorios','rdc']).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
    
    
    
    
    
    
    