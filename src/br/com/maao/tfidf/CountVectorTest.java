package br.com.maao.tfidf;

import java.util.List;

import org.junit.Assert;
import org.junit.Test;

public class CountVectorTest {
    @Test
    public void testTokenise() {
        var input = List.of("This is the first document",
        "This document is the second document",
        "And this is the third one",
        "Is this the first document");
        var countVector = new CountVector(false);
        var feature = countVector.tokenise(input);

        var document = feature.getDocument(0);
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("this"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("is"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("the"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("first"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("document"));

        document = feature.getDocument(1);
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("this"));
        Assert.assertEquals(Integer.valueOf(2), document.termFrequency.get("document"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("is"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("the"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("second"));

        document = feature.getDocument(2);
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("and"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("this"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("is"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("the"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("third"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("one"));

        document = feature.getDocument(3);
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("is"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("this"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("the"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("first"));
        Assert.assertEquals(Integer.valueOf(1), document.termFrequency.get("document"));

        var vocabulary = feature.getVocabulary();
        Assert.assertTrue(vocabulary.contains("this"));
        Assert.assertTrue(vocabulary.contains("is"));
        Assert.assertTrue(vocabulary.contains("the"));
        Assert.assertTrue(vocabulary.contains("first"));
        Assert.assertTrue(vocabulary.contains("document"));
        Assert.assertTrue(vocabulary.contains("second"));
        Assert.assertTrue(vocabulary.contains("and"));
        Assert.assertTrue(vocabulary.contains("third"));
        Assert.assertTrue(vocabulary.contains("one"));
        Assert.assertFalse(vocabulary.contains("sample"));
    }
}
