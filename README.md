# TF-IDF Similarity Calculator

Este é um projeto simples em Java que implementa um sistema de recomendação baseado em similaridade de documentos utilizando **TF-IDF (Term Frequency-Inverse Document Frequency)**. O projeto permite calcular a similaridade entre documentos usando **cosine similarity** e **dot product**.

## Funcionalidades

- **Tokenização**: Realiza a tokenização básica dos textos para segmentar as palavras.
- **TF-IDF**: Calcula o peso TF-IDF de cada termo em cada documento.
- **Métricas de Similaridade**:
  - **Cosine Similarity**: Mede a similaridade de cosseno entre dois documentos, com valores entre -1 e 1.
  - **Dot Product**: Mede o produto escalar entre os vetores TF-IDF de dois documentos, retornando um valor numérico.

## Como Funciona

O algoritmo converte cada documento em um vetor TF-IDF e calcula a similaridade entre pares de documentos. Ele permite encontrar documentos que compartilham temas e identificar conteúdos semelhantes no conjunto.

### Exemplo de Uso

```java
var input = List.of("This is the first document",
                "This document is the second document",
                "And this is the third one",
                "Is this the first document");
var tfidf = new TfidfVectorizer(false, true, false, true);
var similarityCalculator = new SimilarityCalculator(tfidf.fitTransform(input));

double cosineSim = similarityCalculator.cosine(0, 1);
double dotProd = similarityCalculator.dotProduct(0, 1);
int mostSimilar = similarityCalculator.findMostSimilar(1);

System.out.println("Cosine Similarity: " + cosineSim);
System.out.println("Dot Product: " + dotProd);
System.out.println("Found most similar (1): " + mostSimilar);
```

## TODO: 
* stopwords
* stemming
