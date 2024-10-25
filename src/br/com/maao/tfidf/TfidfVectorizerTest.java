package br.com.maao.tfidf;

import java.util.List;

import org.junit.Assert;
import org.junit.Test;

public class TfidfVectorizerTest {

    @Test
    public void testFitTransform() {

        var input = List.of("This is the first document",
                "This document is the second document",
                "And this is the third one",
                "Is this the first document");
        var tfidf = new TfidfVectorizer(false, true, false, true);
        tfidf.fitTransform(input);
        var docs = tfidf.getFeatureDocuments();

        /**
         * Calculate TF (Term Frequency)
         */
        Assert.assertEquals(0.2, docs.get(0).tf.get("this"), .01);
        Assert.assertEquals(0.2, docs.get(0).tf.get("is"), .01);
        Assert.assertEquals(0.2, docs.get(0).tf.get("the"), .01);
        Assert.assertEquals(0.2, docs.get(0).tf.get("first"), .01);
        Assert.assertEquals(0.2, docs.get(0).tf.get("document"), .01);

        Assert.assertEquals(0.167, docs.get(1).tf.get("this"), .01);
        Assert.assertEquals(0.167, docs.get(1).tf.get("is"), .01);
        Assert.assertEquals(0.167, docs.get(1).tf.get("the"), .01);
        Assert.assertEquals(0.333, docs.get(1).tf.get("document"), .01);
        Assert.assertEquals(0.17, docs.get(1).tf.get("second"), .01);

        Assert.assertEquals(0.167, docs.get(2).tf.get("this"), .01);
        Assert.assertEquals(0.167, docs.get(2).tf.get("is"), .01);
        Assert.assertEquals(0.167, docs.get(2).tf.get("the"), .01);
        Assert.assertEquals(0.17, docs.get(2).tf.get("and"), .01);
        Assert.assertEquals(0.17, docs.get(2).tf.get("third"), .01);
        Assert.assertEquals(0.17, docs.get(2).tf.get("one"), .01);

        Assert.assertEquals(0.2, docs.get(3).tf.get("is"), .01);
        Assert.assertEquals(0.2, docs.get(3).tf.get("this"), .01);
        Assert.assertEquals(0.2, docs.get(3).tf.get("the"), .01);
        Assert.assertEquals(0.2, docs.get(3).tf.get("first"), .01);
        Assert.assertEquals(0.2, docs.get(3).tf.get("document"), .01);

        /**
         * Calculate IDF (Inverse Document Frequency)
         */
        Assert.assertEquals(1, tfidf.idf("this"), .01);
        Assert.assertEquals(1, tfidf.idf("is"), .01);
        Assert.assertEquals(1, tfidf.idf("the"), .01);
        Assert.assertEquals(1.51, tfidf.idf("first"), .01);
        Assert.assertEquals(1.22, tfidf.idf("document"), .01);
        Assert.assertEquals(1.91, tfidf.idf("second"), .01);
        Assert.assertEquals(1.91, tfidf.idf("and"), .01);
        Assert.assertEquals(1.91, tfidf.idf("third"), .01);
        Assert.assertEquals(1.91, tfidf.idf("one"), .01);

        /**
         * L2 Normalization
         */
        Assert.assertEquals(0.520717, docs.get(0).l2, .01);
        Assert.assertEquals(0.592932, docs.get(1).l2, .01);
        Assert.assertEquals(0.623977, docs.get(2).l2, .01);
        Assert.assertEquals(0.520717, docs.get(3).l2, .01);

        /**
         * Calculate TF * IDF
         */
        Assert.assertEquals(0.384, docs.get(0).tfIdf.get("this"), .01);
        Assert.assertEquals(0.384, docs.get(0).tfIdf.get("is"), .01);
        Assert.assertEquals(0.384, docs.get(0).tfIdf.get("the"), .01);
        Assert.assertEquals(0.58, docs.get(0).tfIdf.get("first"), .01);
        Assert.assertEquals(0.4697, docs.get(0).tfIdf.get("document"), .01);

        Assert.assertEquals(0.281, docs.get(1).tfIdf.get("this"), .01);
        Assert.assertEquals(0.281, docs.get(1).tfIdf.get("is"), .01);
        Assert.assertEquals(0.281, docs.get(1).tfIdf.get("the"), .01);
        Assert.assertEquals(0.6876, docs.get(1).tfIdf.get("document"), .01);
        Assert.assertEquals(0.5386, docs.get(1).tfIdf.get("second"), .01);

        Assert.assertEquals(0.267, docs.get(2).tfIdf.get("this"), .01);
        Assert.assertEquals(0.267, docs.get(2).tfIdf.get("is"), .01);
        Assert.assertEquals(0.267, docs.get(2).tfIdf.get("the"), .01);
        Assert.assertEquals(0.51, docs.get(2).tfIdf.get("and"), .01);
        Assert.assertEquals(0.51, docs.get(2).tfIdf.get("third"), .01);
        Assert.assertEquals(0.51, docs.get(2).tfIdf.get("one"), .01);

        Assert.assertEquals(0.384, docs.get(3).tfIdf.get("is"), .01);
        Assert.assertEquals(0.384, docs.get(3).tfIdf.get("this"), .01);
        Assert.assertEquals(0.384, docs.get(3).tfIdf.get("the"), .01);
        Assert.assertEquals(0.58, docs.get(3).tfIdf.get("first"), .01);
        Assert.assertEquals(0.4697, docs.get(3).tfIdf.get("document"), .01);

    }

    @Test
    public void testProductEscalar() {
        Double[][] matrix = {
                { 0.5, 0.2, 0.0, 0.82 },
                { 0.3, 0.2, 0.4, 0.5 }
        };
        var cal = new SimilarityCalculator(matrix);
        Assert.assertEquals(0.59, cal.dotProduct(0, 1), 0.1);
    }

    @Test
    public void testSimilaridadeCosseno() {
        Double[][] matrix = {
                { 0.5, 0.2, 0.0, 0.82 },
                { 0.3, 0.2, 0.4, 0.5 }
        };
        var cal = new SimilarityCalculator(matrix);
        Assert.assertEquals(0.832, cal.cosine(0, 1), 0.1);
    }

    @Test
    public void testMostSimilar() {
        var input = List.of("This is the first document",
                "This document is the second document",
                "And this is the third one",
                "Is this the first document");
        var tfidf = new TfidfVectorizer(false, true, false, true);
        var cal = new SimilarityCalculator(tfidf.fitTransform(input));
        int result = cal.findMostSimilar(1);
        Assert.assertEquals(0, result);

    }

    @Test
    public void testMostSimilarNotFound() {
        var input = List.of("This is the first document",
                "This document is the second document",
                "And this is the third one",
                "Is this the first document");
        var tfidf = new TfidfVectorizer(false, true, false, true);
        var cal = new SimilarityCalculator(tfidf.fitTransform(input));
        int result = cal.findMostSimilar(2);
        Assert.assertEquals(-1, result);

    }
}
