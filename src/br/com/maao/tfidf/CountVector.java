package br.com.maao.tfidf;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.StringTokenizer;

public class CountVector {

    final List<String> stopWords = List.of("a", "com", "está", "muitas");

    private final boolean useStopWorkds;
    
    public CountVector() {
        this.useStopWorkds = true;
    }

    public CountVector(boolean useStopWorkds) {
        this.useStopWorkds = useStopWorkds;
    }

    private List<String> tokenise(String doc) {
        List<String> tokens = new ArrayList<>();
        StringTokenizer tokenizer = new StringTokenizer(doc);
        while (tokenizer.hasMoreElements()) {
            tokens.add(tokenizer.nextToken());
        }
        return tokens;
    }


     protected Feature tokenise(List<String> documents) {
        final Feature feature = new Feature();
        for (String document: documents) {
            var values = tokenise(document);
            List<String> terms = new ArrayList<>();
            LinkedHashMap<String, Integer>  termCount = new LinkedHashMap<>();
            for (String token: values) {
                var tmp = token.trim().toLowerCase();
                if (tmp.isEmpty() || (useStopWorkds && stopWords.contains(tmp))) {
                    continue;
                }

                feature.addVocabulary(tmp);

                terms.add(tmp);
                termCount.put(tmp, termCount.get(tmp) == null? 1: termCount.get(tmp) + 1);
            }
            feature.addDocument(new Document(document, terms, termCount));
        }
        
        return feature;
    }



}
