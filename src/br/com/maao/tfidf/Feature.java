package br.com.maao.tfidf;

import java.util.List;
import java.util.ArrayList;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.Iterator;

public class Feature {

    private SortedSet<String> vocabulary = new TreeSet<>(String.CASE_INSENSITIVE_ORDER);

    private List<Document> documents = new ArrayList<>();

    public Double[][] vec2d() {
        Double[][] matrix = new Double[this.documents.size()][vocabulary.size()];
        for (int i = 0; i < this.documents.size(); i++) {
            var document = this.documents.get(i);
            int j = 0;
            for (Iterator<String> it = this.vocabulary.iterator(); it.hasNext(); j++) {
                double tfidf = 0;
                String key = it.next();
                if (document.tfIdf.containsKey(key)) {
                    tfidf = document.tfIdf.get(key);
                }
                matrix[i][j] = tfidf;
            }
        }
        return matrix;
    }

    public void addVocabulary(String term) {
        this.vocabulary.add(term);
    }

    public void addDocument(Document document) {
        this.documents.add(document);
    }

    public Document getDocument(int index) {
        return this.documents.get(index);
    }

    public SortedSet<String> getVocabulary() {
        return vocabulary;
    }

    public List<Document> getDocuments() {
        return documents;
    }

    public int sizeDocuments() {
        return this.documents.size();
    }

    public int sizeVocabulary() {
        return this.vocabulary.size();
    }

    public List<String> getListVocabulary() {
        return new ArrayList<>(vocabulary);
    }
    
}
