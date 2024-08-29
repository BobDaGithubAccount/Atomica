package net.atomica.engine.shaders;

import org.lwjgl.opengl.GL20;
import org.lwjgl.opengl.GL30;
import org.lwjgl.opengl.GL15;
import org.lwjgl.system.MemoryStack;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.HashMap;
import java.util.Map;

public abstract class Shader {
    protected int programId;
    protected int vaoId;
    protected final Map<String, Integer> uniformLocations;

    public Shader() {
        programId = GL20.glCreateProgram();
        if (programId == 0) {
            throw new RuntimeException("Could not create Shader");
        }
        uniformLocations = new HashMap<>();
        vaoId = GL30.glGenVertexArrays();
    }

    public abstract void bind();

    public abstract void unbind();

    public abstract void dispose();

    public abstract void setUniform1i(String name, int value);

    public abstract void setUniform1f(String name, float value);

    public abstract void setUniformMatrix4fv(String name, FloatBuffer matrix);

    public abstract void setUniform3f(String name, float x, float y, float z);

    public abstract void setUniform4f(String name, float x, float y, float z, float w);

    public abstract void setUniform2f(String name, float x, float y);

    protected int getUniformLocation(String name) {
        if (uniformLocations.containsKey(name)) {
            return uniformLocations.get(name);
        }
        int location = GL20.glGetUniformLocation(programId, name);
        if (location < 0) {
            System.err.println("Uniform location not found for: " + name);
        } else {
            uniformLocations.put(name, location);
        }
        return location;
    }

    protected void compileShader(String shaderCode, int shaderType) {
        int shaderId = GL20.glCreateShader(shaderType);
        if (shaderId == 0) {
            throw new RuntimeException("Error creating shader. Type: " + shaderType);
        }

        GL20.glShaderSource(shaderId, shaderCode);
        GL20.glCompileShader(shaderId);

        if (GL20.glGetShaderi(shaderId, GL20.GL_COMPILE_STATUS) == GL20.GL_FALSE) {
            throw new RuntimeException("Error compiling Shader code: " + GL20.glGetShaderInfoLog(shaderId, 1024));
        }

        GL20.glAttachShader(programId, shaderId);
    }

    protected void link() {
        GL20.glLinkProgram(programId);
        if (GL20.glGetProgrami(programId, GL20.GL_LINK_STATUS) == GL20.GL_FALSE) {
            throw new RuntimeException("Error linking Shader code: " + GL20.glGetProgramInfoLog(programId, 1024));
        }

        GL20.glValidateProgram(programId);
        if (GL20.glGetProgrami(programId, GL20.GL_VALIDATE_STATUS) == GL20.GL_FALSE) {
            System.err.println("Warning validating Shader code: " + GL20.glGetProgramInfoLog(programId, 1024));
        }
    }

    protected void detachAndDeleteShader(int shaderId) {
        GL20.glDetachShader(programId, shaderId);
        GL20.glDeleteShader(shaderId);
    }

    public void setupVertexAttribPointer(int index, int size, int stride, int offset) {
        GL20.glEnableVertexAttribArray(index);
        GL20.glVertexAttribPointer(index, size, GL20.GL_FLOAT, false, stride, offset);
    }

    public void bindVAO() {
        GL30.glBindVertexArray(vaoId);
    }

    public void unbindVAO() {
        GL30.glBindVertexArray(0);
    }

    public void createVBO(float[] data, int attributeNumber, int size, int stride, int offset) {
        int vboId = GL15.glGenBuffers();
        GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, vboId);
        try (MemoryStack stack = MemoryStack.stackPush()) {
            FloatBuffer buffer = stack.callocFloat(data.length);
            buffer.put(data).flip();
            GL15.glBufferData(GL15.GL_ARRAY_BUFFER, buffer, GL15.GL_STATIC_DRAW);
        }
        setupVertexAttribPointer(attributeNumber, size, stride, offset);
        GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, 0);
    }

    public void createEBO(int[] indices) {
        int eboId = GL15.glGenBuffers();
        GL15.glBindBuffer(GL15.GL_ELEMENT_ARRAY_BUFFER, eboId);
        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer buffer = stack.callocInt(indices.length);
            buffer.put(indices).flip();
            GL15.glBufferData(GL15.GL_ELEMENT_ARRAY_BUFFER, buffer, GL15.GL_STATIC_DRAW);
        }
    }
}
