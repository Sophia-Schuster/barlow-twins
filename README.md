# Barlow Twins e Rede Siamesa

 Nos baseamos na arquitetura Barlow-Twins (https://github.com/facebookresearch/barlowtwins) que utiliza da Aprendizagem Auto-Supervisionada com o objetivo de aprender incorporações invariantes à distorção da amostra de entrada. Para isso aplicamos aumentos a amostras de entrada e dirigimos suas representações o mais próximo possível. 
 
 ![image](https://user-images.githubusercontent.com/60801559/230993543-0fdb7872-cc76-40af-94b1-e343cc4becfe.png)

  Fora o Barlow- Twins, também nos baseamos na Rede Siamesa, redes neurais que compartilham pesos entre duas ou mais redes irmãs, cada uma produzindo vetores de incorporação de suas respectivas entradas. Essa rede permiti com que utilizemos duas séries temporais na mesma rede neural simultaneamente.
 Para isso nós criamos pares de imagens (no nosso caso aumentadas como esperado no código do Barlow Twins), rotulamos esses pares como falsos ou verdadeiros. Então nós definimos o modelo da rede, existem duas camadas de entrada, cada uma levando à sua própria rede, que produz embeddings. Uma camada Lambda então os mescla usando uma distância euclidiana e a saída mesclada é alimentada na rede final. Por fim compilamos o modelo final utilizando perda contrastiva e depois podemos seguir para o treinamento e avaliação normalmente. 
 

## Preparação 
Alguns exemplos de datasets do Physionet aplicaveis para treinamento ou avaliação (ECG 128 p/second):

https://physionet.org/content/shareedb/ (52464 * 2 samples, a cada 1 conjunto)
https://physionet.org/content/aftdb/ (3748 * 2 samples todos conjuntos)
https://physionet.org/content/apnea-ecg/1.0.0/ (79729 * 2 samples, a cada 5 conjuntos) 

## Treinar e avaliar Decoder

Para gerar e treinar e avaliar seu modelo basta salvar o dataset na pasta ./data, no momento estão salvos 5 subconjuntos do dataset https://physionet.org/content/apnea-ecg/1.0.0/ e executar:

```bash
python btCode.py
``` 

# Avaliar Encoder

Para avaliar seu modelo de Encoder você deve antes passar pela etapa anterior, então sera gerado um arquivo "encoder_model.h5"deixe ele na pasta principal onde foi gerado, salve o dataset desejado para se obter features na pasta ./data, por exemplo outros 5 subconjuntos do dataset https://physionet.org/content/apnea-ecg/1.0.0/ e execute:


```bash
python btCodeWM.py 
``` 

Sera então gerados o encoder.txt (muito grande para ser colocado de exemplo no git) e description.txt
