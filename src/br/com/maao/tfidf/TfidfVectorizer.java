package br.com.maao.tfidf;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TfidfVectorizer {

    private Feature feature;

    private Map<String, Double> idf = new HashMap<>();

    private final CountVector countVector;
    private final boolean smoothIdf;
    private final boolean sublinearTf;
    private final boolean l2Normalization;

    public TfidfVectorizer() {
        this.countVector = new CountVector();
        this.smoothIdf = false;
        this.sublinearTf = false;
        this.l2Normalization = false;
    }

    public TfidfVectorizer(boolean useStopWorkds) {
        this.countVector = new CountVector(useStopWorkds);
        this.smoothIdf = false;
        this.sublinearTf = false;
        this.l2Normalization = false;
    }

    public TfidfVectorizer(boolean useStopWorkds, boolean smoothIdf, boolean sublinearTf, boolean l2Normalization) {
        this.countVector = new CountVector(useStopWorkds);
        this.smoothIdf = smoothIdf;
        this.sublinearTf = sublinearTf;
        this.l2Normalization = l2Normalization;
    }

    public Double[][] fitTransform(List<String> documents) {
        this.feature = this.countVector.tokenise(documents);

        this.calculateTF();

        this.idf = this.calculateIDF();

        this.calculateTfIdf();

        return this.feature.vec2d();
    }

    private void calculateTfIdf() {
        for (Document document : this.feature.getDocuments()) {
            for (String term : document.tf.keySet()) {
                var tf = document.tf.get(term);
                var idf = this.idf.get(term);
                var result = tf * idf;
                document.tfIdf.put(term, result);
            }
        }
        if (l2Normalization) {
            l2Normalization();
        }
    }

    private void l2Normalization() {
        for (Document document : this.feature.getDocuments()) {
            double acumulate = 0.0;
            for (double value : document.tfIdf.values()) {
                acumulate += (Math.pow(value, 2));
            }
            double l2 = Math.sqrt(acumulate);
            document.l2 = l2;
            for (String term : document.tfIdf.keySet()) {
                double previousL2 = document.tfIdf.get(term);
                double afterL2 = previousL2 / l2;
                document.tfIdf.put(term, afterL2);
            }
        }
    }

    private void calculateTF() {
        for (Document document : this.feature.getDocuments()) {
            calculateTF(document);
        }
    }

    private void calculateTF(Document document) {
        for (String term : document.termFrequency.keySet()) {
            var dFreq = document.termFrequency.get(term);
            var count = document.terms.size();
            var result = Double.valueOf(dFreq) / Double.valueOf(count);
            if (sublinearTf) {
                result = 1 + Math.log(result);
            }
            document.tf.put(term, result);
        }
    }

    private Map<String, Double> calculateIDF() {
        final int numberOfDocuments = this.feature.sizeDocuments();
        final Map<String, Double> result = new HashMap<>();
        for (String keyVocabulary : this.feature.getVocabulary()) {
            int countTermInDocument = 0;
            for (Document document : this.feature.getDocuments()) {
                if (document.termFrequency.containsKey(keyVocabulary)) {
                    countTermInDocument += 1;
                }
            }
            double idf = 0.0;
            if (smoothIdf) {
                double temp = Double.valueOf(numberOfDocuments + 1) / Double.valueOf(countTermInDocument + 1);
                idf = 1.0 + Math.log(temp);
            } else {
                double temp = Double.valueOf(numberOfDocuments) / Double.valueOf(countTermInDocument);
                idf = Math.log(temp);
            }
            result.put(keyVocabulary, idf);

        }
        return result;
    }

    public Double idf(String key) {
        return idf.get(key);
    }

    List<String> getFeatureNamesOut() {
        return this.feature.getListVocabulary();
    }

    List<Document> getFeatureDocuments() {
        return this.feature.getDocuments();
    }

}
