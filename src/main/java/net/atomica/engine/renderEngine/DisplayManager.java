package net.atomica.engine.renderEngine;

import net.atomica.logging.Logger;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWVidMode;
import org.lwjgl.opengl.GL;
import org.lwjgl.opengl.GL20;
import org.lwjgl.system.MemoryUtil;

public class DisplayManager {

	public static final int WIDTH = 1280;
	public static final int HEIGHT = 720;
	public static final int FPS_CAP = 60;
	public static final String TITLE = "Atomica";

	public long window;

	public void createDisplay() {
		if (!GLFW.glfwInit()) {
			Logger.log("Unable to initialize GLFW");
			throw new IllegalStateException("Unable to initialize GLFW");
		}

		window = GLFW.glfwCreateWindow(WIDTH, HEIGHT, TITLE, MemoryUtil.NULL, MemoryUtil.NULL);
		if (window == MemoryUtil.NULL) {
			Logger.log("Failed to create the GLFW window");
			throw new RuntimeException("Failed to create the GLFW window");
		}

		GLFW.glfwMakeContextCurrent(window);
		GLFW.glfwSwapInterval(1);
		GLFW.glfwShowWindow(window);

		GL.createCapabilities();
		GL.createCapabilities();

		GL.createCapabilities();
		GL20.glViewport(0, 0, WIDTH, HEIGHT);
	}

	public void updateDisplay() {
		GLFW.glfwSwapBuffers(window);
		GLFW.glfwPollEvents();

		try {
			Thread.sleep(1000 / FPS_CAP);
		} catch (InterruptedException e) {
			Logger.log(e.getMessage());
			e.printStackTrace();
		}
	}

	public void closeDisplay() {
		GLFW.glfwDestroyWindow(window);
		GLFW.glfwTerminate();
	}

	public void updateTitleWithFPS(int fps) {
		GLFW.glfwSetWindowTitle(window, TITLE + " | FPS: " + fps);
	}

	public boolean shouldClose() {
		return GLFW.glfwWindowShouldClose(window);
	}
}
