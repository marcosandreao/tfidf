package br.com.maao.tfidf;

import java.util.LinkedHashMap;
import java.util.List;

public class Document {
    public String rawDocument;
    public List<String> terms;
    public LinkedHashMap<String, Integer> termFrequency = new LinkedHashMap<>();
    public LinkedHashMap<String, Double> tf = new LinkedHashMap<>();
    public LinkedHashMap<String, Double> tfIdf = new LinkedHashMap<>();
    public double l2;
    
    public Document(String document, List<String> terms) {
        this.rawDocument = document;
        this.terms = terms;
    }

    public Document(String document, List<String> terms, LinkedHashMap<String, Integer> termFrequency) {
        this(document, terms);
        this.termFrequency = termFrequency;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((rawDocument == null) ? 0 : rawDocument.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Document other = (Document) obj;
        if (rawDocument == null) {
            if (other.rawDocument != null)
                return false;
        } else if (!rawDocument.equals(other.rawDocument))
            return false;
        return true;
    }

}
