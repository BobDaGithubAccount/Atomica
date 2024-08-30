package net.atomica.engine.render;

import net.atomica.engine.shaders.ShaderManager;
import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.lwjgl.BufferUtils;

import java.nio.FloatBuffer;

public class Model {
    public Mesh mesh;
    public Texture texture;
    public Vector3f position;
    public Vector3f rotation;
    public Vector3f scale;

    public Model(Mesh mesh, Texture texture, Vector3f position, Vector3f rotation, Vector3f scale) {
        this.mesh = mesh;
        this.texture = texture;
        this.position = position;
        this.rotation = rotation;
        this.scale = scale;
    }

    public void render(ShaderManager shader) {
        Matrix4f modelMatrix = new Matrix4f()
                .translate(position)
                .rotateX((float) Math.toRadians(rotation.x))
                .rotateY((float) Math.toRadians(rotation.y))
                .rotateZ((float) Math.toRadians(rotation.z))
                .scale(scale);

        FloatBuffer modelBuffer = BufferUtils.createFloatBuffer(16);
        modelMatrix.get(modelBuffer);
        shader.setUniformMatrix4fv("model", modelBuffer);

        texture.bind();
        mesh.render();
        texture.unbind();
    }

    public void dispose() {
        mesh.dispose();
        texture.dispose();
    }
}
