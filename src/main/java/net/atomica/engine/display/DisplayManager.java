package net.atomica.engine.display;

import org.lwjgl.glfw.*;
import org.lwjgl.opengl.GL;
import org.lwjgl.opengl.GL11;

public class DisplayManager {
    private long window;
    private int width, height;
    private String title;
    private boolean isFullscreen;
    private GLFWVidMode videoMode;

    public DisplayManager(int width, int height, String title) {
        this.width = width;
        this.height = height;
        this.title = title;
        this.isFullscreen = false;
    }

    public void createDisplay() {
        if (!GLFW.glfwInit()) {
            throw new IllegalStateException("Failed to initialize GLFW");
        }

        GLFW.glfwWindowHint(GLFW.GLFW_CONTEXT_VERSION_MAJOR, 3);
        GLFW.glfwWindowHint(GLFW.GLFW_CONTEXT_VERSION_MINOR, 3);
        GLFW.glfwWindowHint(GLFW.GLFW_OPENGL_PROFILE, GLFW.GLFW_OPENGL_CORE_PROFILE);

        videoMode = GLFW.glfwGetVideoMode(GLFW.glfwGetPrimaryMonitor());

        window = GLFW.glfwCreateWindow(width, height, title, 0, 0);
        if (window == 0) {
            throw new RuntimeException("Failed to create window");
        }

        centerWindow();

        GLFW.glfwMakeContextCurrent(window);
        GLFW.glfwShowWindow(window);
        GL.createCapabilities();
        GL11.glEnable(GL11.GL_DEPTH_TEST);

        GLFW.glfwSetFramebufferSizeCallback(window, (window, newWidth, newHeight) -> {
            this.width = newWidth;
            this.height = newHeight;
            GL11.glViewport(0, 0, newWidth, newHeight);
        });
    }

    public void updateDisplay() {
        GLFW.glfwSwapBuffers(window);
        GLFW.glfwPollEvents();
    }

    public boolean shouldClose() {
        return GLFW.glfwWindowShouldClose(window);
    }

    public void closeDisplay() {
        GLFW.glfwDestroyWindow(window);
        GLFW.glfwTerminate();
    }

    public void centerWindow() {
        if (!isFullscreen) {
            GLFWVidMode vidmode = videoMode != null ? videoMode : GLFW.glfwGetVideoMode(GLFW.glfwGetPrimaryMonitor());
            GLFW.glfwSetWindowPos(
                    window,
                    (vidmode.width() - width) / 2,
                    (vidmode.height() - height) / 2
            );
        }
    }

    public void toggleFullscreen() {
        isFullscreen = !isFullscreen;

        if (isFullscreen) {
            GLFW.glfwSetWindowMonitor(window, GLFW.glfwGetPrimaryMonitor(), 0, 0, videoMode.width(), videoMode.height(), videoMode.refreshRate());
        } else {
            GLFW.glfwSetWindowMonitor(window, 0, 100, 100, width, height, videoMode.refreshRate());
            centerWindow();
        }

        GL11.glViewport(0, 0, videoMode.width(), videoMode.height());
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public long getWindow() {
        return window;
    }
}
