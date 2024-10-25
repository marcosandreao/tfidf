package br.com.maao.tfidf;

public class SimilarityCalculator {

    private static double kMinSimilarity = 0.5;
    private final Double[][] tfIdfVec2D;

    private double minSimilarity;

    public SimilarityCalculator(Double[][] tfIdfVec2D) {
        this.tfIdfVec2D = tfIdfVec2D;
        this.minSimilarity =  kMinSimilarity;
    }

    public SimilarityCalculator(Double[][] tfIdfVec2D, double minSimilarity) {
        this(tfIdfVec2D);
        this.minSimilarity = minSimilarity;
    }

    private double magnitude(Double[] values) {
        double result = 0.0;
        for (double val : values) {
            result += Math.pow(val, 2);
        }
        return Math.sqrt(result);
    }

    private double dotProduct(Double[] d1, Double[] d2) {
        double result = 0.0;
        for (int i = 0; i < d1.length; ++i) {
            result += (d1[i] * d2[i]);
        }
        return result;
    }

    public double dotProduct(int indexD1, int indexD2) {
        return dotProduct(this.tfIdfVec2D[indexD1], this.tfIdfVec2D[indexD2]);
    }

    public double cosine(Double[] d1, Double[] d2) {
        return dotProduct(d1, d2) / (magnitude(d1) * magnitude(d2));
    }

    public double cosine(int indexD1, int indexD2) {
        return dotProduct(this.tfIdfVec2D[indexD1], this.tfIdfVec2D[indexD2])
                / (magnitude(this.tfIdfVec2D[indexD1]) * magnitude(this.tfIdfVec2D[indexD2]));
    }

    public int findMostSimilar(int targetIndex) {
        int mostSimilarIndex = -1;
        double lastSimilarity = minSimilarity;
        for (int i = 0; i < this.tfIdfVec2D.length; ++i) {
            if (i == targetIndex) {
                continue;
            }
            double similarity = cosine(this.tfIdfVec2D[targetIndex], this.tfIdfVec2D[i]);
            if (similarity > lastSimilarity) {
                lastSimilarity = similarity;
                mostSimilarIndex = i;
            }
        }

        return mostSimilarIndex;
    }

}
