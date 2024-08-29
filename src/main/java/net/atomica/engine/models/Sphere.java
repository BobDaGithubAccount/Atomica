package net.atomica.engine.models;

import java.util.ArrayList;
import java.util.List;

public class Sphere {
    public static float[] generateVertices(int latitudeBands, int longitudeBands, float radius) {
        List<Float> vertices = new ArrayList<>();

        for (int lat = 0; lat <= latitudeBands; lat++) {
            double theta = lat * Math.PI / latitudeBands;
            double sinTheta = Math.sin(theta);
            double cosTheta = Math.cos(theta);

            for (int lon = 0; lon <= longitudeBands; lon++) {
                double phi = lon * 2 * Math.PI / longitudeBands;
                double sinPhi = Math.sin(phi);
                double cosPhi = Math.cos(phi);

                float x = (float) (cosPhi * sinTheta);
                float y = (float) cosTheta;
                float z = (float) (sinPhi * sinTheta);
                float u = (float) (lon / (double) longitudeBands);
                float v = (float) (lat / (double) latitudeBands);

                vertices.add(x * radius);
                vertices.add(y * radius);
                vertices.add(z * radius);
                vertices.add(x);
                vertices.add(y);
                vertices.add(z);
                vertices.add(u);
                vertices.add(v);
            }
        }

        float[] vertexData = new float[vertices.size()];
        for (int i = 0; i < vertices.size(); i++) {
            vertexData[i] = vertices.get(i);
        }

        return vertexData;
    }

    public static int[] generateIndices(int latitudeBands, int longitudeBands) {
        List<Integer> indices = new ArrayList<>();

        for (int lat = 0; lat < latitudeBands; lat++) {
            for (int lon = 0; lon < longitudeBands; lon++) {
                int first = (lat * (longitudeBands + 1)) + lon;
                int second = first + longitudeBands + 1;

                indices.add(first);
                indices.add(second);
                indices.add(first + 1);

                indices.add(second);
                indices.add(second + 1);
                indices.add(first + 1);
            }
        }

        int[] indexData = new int[indices.size()];
        for (int i = 0; i < indices.size(); i++) {
            indexData[i] = indices.get(i);
        }

        return indexData;
    }
}
