package net.atomica.engine.render;

import net.atomica.Main;
import net.atomica.engine.display.DisplayManager;
import net.atomica.engine.logic.KeyPressEvent;
import net.atomica.engine.models.Sphere;
import net.atomica.engine.shaders.ShaderManager;
import net.atomica.engine.logic.EventManager;
import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.lwjgl.BufferUtils;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWKeyCallback;
import org.lwjgl.opengl.GL11;

import java.io.BufferedReader;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class RenderEngine {
    public ShaderManager shader;
    public Camera camera;
    public DisplayManager displayManager;

    public RenderEngine() {
        this.displayManager = new DisplayManager(1280, 720, "Atomica");
        displayManager.createDisplay();
        this.camera = new Camera(new Vector3f(0, 0, 3), 0, 0, 0);
        this.shader = new ShaderManager(loadShader("vertex_shader"), loadShader("fragment_shader"));
        GLFW.glfwSetKeyCallback(displayManager.getWindow(), new GLFWKeyCallback() {
            @Override
            public void invoke(long window, int key, int scancode, int action, int mods) {
                if (action == GLFW.GLFW_PRESS) {
                    EventManager.fireEvent(new KeyPressEvent(key));
                }
            }
        });
    }

    public void run() {
        int latitudeBands = 30;
        int longitudeBands = 30;
        float radius = 1.0f;

        float[] vertices = Sphere.generateVertices(latitudeBands, longitudeBands, radius);
        int[] indices = Sphere.generateIndices(latitudeBands, longitudeBands);

        Mesh mesh = new Mesh(vertices, indices);
        Texture texture = new Texture(Main.assetsDir.getPath() + "/textures/essential.png");
        Model sphereModel = new Model(mesh, texture, new Vector3f(0, 0, 0), new Vector3f(0, 0, 0), new Vector3f(1, 1, 1));

        while (!displayManager.shouldClose()) {

            clear();

            Matrix4f projectionMatrix = new Matrix4f().perspective(
                    (float) Math.toRadians(60.0f),//FOV
                    (float) displayManager.getWidth() / displayManager.getHeight(),
                    0.1f,
                    1000f
            );

            sphereModel.rotation.y += 0.1f;

            List<Model> models = new ArrayList<>();
            models.add(sphereModel);

            render(models, projectionMatrix);

            displayManager.updateDisplay();
        }

        sphereModel.dispose();
        shader.dispose();
        displayManager.closeDisplay();
    }

    public void render(List<Model> models, Matrix4f projectionMatrix) {
        shader.bind();

        FloatBuffer viewBuffer = BufferUtils.createFloatBuffer(16);
        camera.getViewMatrix().get(viewBuffer);
        shader.setUniformMatrix4fv("view", viewBuffer);

        FloatBuffer projectionBuffer = BufferUtils.createFloatBuffer(16);
        projectionMatrix.get(projectionBuffer);
        shader.setUniformMatrix4fv("projection", projectionBuffer);

//        shader.setUniform3f("lightPos", 0.0f, 10.0f, 0.0f);
//        shader.setUniform3f("viewPos", camera.position.x, camera.position.y, camera.position.z);
//        shader.setUniform3f("lightColor", 1.0f, 1.0f, 1.0f);

        for (Model model : models) {
            model.render(shader);
        }

        shader.unbind();
    }

    public static void clear() {
        GL11.glClear(GL11.GL_COLOR_BUFFER_BIT | GL11.GL_DEPTH_BUFFER_BIT);
        GL11.glClearColor(0.53f, 0.81f, 0.92f, 1.0f);
    }

    private String loadShader(String name) {
        try {
            StringBuilder shaderCode = new StringBuilder();
            Path shaderPath = Paths.get(Main.assetsDir.getPath(), "shaders/", name + ".glsl");
            try (BufferedReader reader = Files.newBufferedReader(shaderPath)) {
                String line;
                while ((line = reader.readLine()) != null) {
                    shaderCode.append(line).append("\n");
                }
            }
            return shaderCode.toString();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        return null;
    }
}
