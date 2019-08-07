# Clusterização das respostas dadas às perguntas de uma consulta dirigida de laboratórios clínicos.

Foi finalizada a abordagem clássica para fazer a clusterização dos textos. Foi feito um forte pré-processamento colocando todas as letras como minúsculas, retirando pontuações, retirando stop words etc. Após o pré-processamento, um bag of words é gerado e é aplicado o tfidf nesse bag of words. Após isso, fazemos uma redução de dimensionalidade usando o Truncated SVD. Por fim, o algoritmo de clusterização escolhido é o hierarchical clustering.

Os resultados obtidos estão dispostos em três arquivos: info_cluster.csv, codigos_perguntas.csv e texto_respostas_por_cluster.csv.

info_cluster.csv: arquivo que contém o código da pergunta, o código da cluster e o número de respostas na cluster.
codigos_perguntas.csv: arquivo que contém o código da pergunta e a pergunta em si.
texto_respostas_por_cluster.csv: arquivo que contém o código da pergunta, o código da cluster, o código da resposta, a resposta sem pré-processamento e a resposta com pré-processamento.
