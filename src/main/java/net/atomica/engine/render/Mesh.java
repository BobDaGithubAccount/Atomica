package net.atomica.engine.render;

import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL15;
import org.lwjgl.opengl.GL20;
import org.lwjgl.opengl.GL30;
import org.lwjgl.system.MemoryUtil;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public class Mesh {
    private final int vaoId;
    private final int vertexCount;
    private final int vboId;
    private final int eboId;

    public Mesh(float[] vertices, int[] indices) {
        vertexCount = indices.length;

        vaoId = GL30.glGenVertexArrays();
        GL30.glBindVertexArray(vaoId);

        vboId = GL15.glGenBuffers();
        GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, vboId);

        FloatBuffer vertexBuffer = MemoryUtil.memAllocFloat(vertices.length);
        vertexBuffer.put(vertices).flip();
        GL15.glBufferData(GL15.GL_ARRAY_BUFFER, vertexBuffer, GL15.GL_STATIC_DRAW);

        GL20.glVertexAttribPointer(0, 3, GL11.GL_FLOAT, false, 8 * Float.BYTES, 0);
        GL20.glEnableVertexAttribArray(0);

        GL20.glVertexAttribPointer(1, 3, GL11.GL_FLOAT, false, 8 * Float.BYTES, 3 * Float.BYTES);
        GL20.glEnableVertexAttribArray(1);

        GL20.glVertexAttribPointer(2, 2, GL11.GL_FLOAT, false, 8 * Float.BYTES, 6 * Float.BYTES);
        GL20.glEnableVertexAttribArray(2);

        eboId = GL15.glGenBuffers();
        GL15.glBindBuffer(GL15.GL_ELEMENT_ARRAY_BUFFER, eboId);

        IntBuffer indicesBuffer = MemoryUtil.memAllocInt(indices.length);
        indicesBuffer.put(indices).flip();
        GL15.glBufferData(GL15.GL_ELEMENT_ARRAY_BUFFER, indicesBuffer, GL15.GL_STATIC_DRAW);

        GL30.glBindVertexArray(0);

        MemoryUtil.memFree(vertexBuffer);
        MemoryUtil.memFree(indicesBuffer);
    }

    public void render() {
        GL30.glBindVertexArray(vaoId);
        GL11.glDrawElements(GL11.GL_TRIANGLES, vertexCount, GL11.GL_UNSIGNED_INT, 0);
        GL30.glBindVertexArray(0);
    }

    public void dispose() {
        GL30.glBindVertexArray(vaoId);
        GL20.glDisableVertexAttribArray(0);

        GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, 0);
        GL15.glDeleteBuffers(vboId);

        GL15.glBindBuffer(GL15.GL_ELEMENT_ARRAY_BUFFER, 0);
        GL15.glDeleteBuffers(eboId);

        GL30.glBindVertexArray(0);
        GL30.glDeleteVertexArrays(vaoId);
    }
}
