package net.atomica.engine.render;

import net.atomica.engine.shaders.ShaderManager;

public class Model {
    private Mesh mesh;
    private Texture texture;

    public Model(Mesh mesh, Texture texture) {
        this.mesh = mesh;
        this.texture = texture;
    }

    public Mesh getMesh() {
        return mesh;
    }

    public Texture getTexture() {
        return texture;
    }

    public void render() {
        texture.bind();
        mesh.render();
        texture.unbind();
    }

    public void dispose() {
        mesh.dispose();
        texture.dispose();
    }
}
