package net.atomica.engine.shaders;

import org.lwjgl.opengl.GL20;
import org.lwjgl.opengl.GL30;
import org.lwjgl.opengl.GL31;

import java.nio.FloatBuffer;

public class ShaderManager extends Shader {

    private int vertexShaderId;
    private int fragmentShaderId;

    public ShaderManager(String vertexShaderCode, String fragmentShaderCode) {
        super();

        vertexShaderId = compile(vertexShaderCode, GL20.GL_VERTEX_SHADER);
        fragmentShaderId = compile(fragmentShaderCode, GL20.GL_FRAGMENT_SHADER);

        link();

        detachAndDeleteShader(vertexShaderId);
        detachAndDeleteShader(fragmentShaderId);
    }

    private int compile(String shaderCode, int type) {
        int shaderId = GL20.glCreateShader(type);
        GL20.glShaderSource(shaderId, shaderCode);
        GL20.glCompileShader(shaderId);
        if (GL20.glGetShaderi(shaderId, GL20.GL_COMPILE_STATUS) == GL20.GL_FALSE) {
            throw new RuntimeException("Error compiling Shader code: " + GL20.glGetShaderInfoLog(shaderId, 1024));
        }
        GL20.glAttachShader(programId, shaderId);
        return shaderId;
    }

    @Override
    public void bind() {
        GL20.glUseProgram(programId);
        bindVAO();
    }

    @Override
    public void unbind() {
        unbindVAO();
        GL20.glUseProgram(0);
    }

    @Override
    public void dispose() {
        unbind();
        if (programId != 0) {
            GL20.glDeleteProgram(programId);
        }
        GL30.glDeleteVertexArrays(vaoId);
    }

    @Override
    public void setUniform1i(String name, int value) {
        GL20.glUniform1i(getUniformLocation(name), value);
    }

    @Override
    public void setUniform1f(String name, float value) {
        GL20.glUniform1f(getUniformLocation(name), value);
    }

    @Override
    public void setUniformMatrix4fv(String name, FloatBuffer matrix) {
        GL20.glUniformMatrix4fv(getUniformLocation(name), false, matrix);
    }

    @Override
    public void setUniform3f(String name, float x, float y, float z) {
        GL20.glUniform3f(getUniformLocation(name), x, y, z);
    }

    @Override
    public void setUniform4f(String name, float x, float y, float z, float w) {
        GL20.glUniform4f(getUniformLocation(name), x, y, z, w);
    }

    @Override
    public void setUniform2f(String name, float x, float y) {
        GL20.glUniform2f(getUniformLocation(name), x, y);
    }

    public void setModelMatrices(FloatBuffer[] modelMatrices) {
        int matrixBufferSize = 4 * 4 * Float.BYTES;
        for (int i = 0; i < modelMatrices.length; i++) {
            int location = GL20.glGetUniformLocation(programId, "model[" + i + "]");
            GL20.glUniformMatrix4fv(location, false, modelMatrices[i]);
        }
    }

    public void drawInstanced(int indexCount, int instanceCount) {
        GL31.glDrawElementsInstanced(GL30.GL_TRIANGLES, indexCount, GL30.GL_UNSIGNED_INT, 0, instanceCount);
    }
}
