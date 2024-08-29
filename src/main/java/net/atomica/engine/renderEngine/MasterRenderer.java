package net.atomica.engine.renderEngine;

import net.atomica.Main;
import net.atomica.engine.entities.Camera;
import net.atomica.engine.entities.Entity;
import net.atomica.engine.entities.Light;
import net.atomica.engine.models.RawModels;
import net.atomica.engine.models.TexturedModel;
import net.atomica.engine.shaders.StaticShader;
import net.atomica.engine.textures.ModelTexture;
import org.joml.Vector3f;
import org.lwjgl.glfw.GLFW;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class MasterRenderer {

	public static StaticShader shader = new StaticShader();
	
	public static Camera camera = new Camera(new Vector3f(0f,0f,0f),0f,0f,0f);
	
	public static Light sun = new Light(new Vector3f(0f,20f,10f), new Vector3f(1f,1f,1f));
	
	private static Map<TexturedModel, List<Entity>> entities = new HashMap<TexturedModel, List<Entity>>();
	
	private static Map<String, RenderObject> objects = new HashMap<String, RenderObject>();
	
//	public static Entity entity = Loader.loadObj("player", new Vector3f(0f,0f,0f), new Vector3f(0f,180f,0f), 0.01f);
//	public static Entity level = Loader.loadLevel(new File("C:\\Users\\duckb\\eclipse-workspace\\Spectre-Client\\server-levels\\localhost.level"));
		
	public static void init() {
		Entity entity = Loader.loadRawModel("essential", RawModels.getErrorCube(), new Vector3f(10f,0f,10f), new Vector3f(0f,180f,0f), 0.01f);
	}
	
	public static void render() {
		Map<String, RenderObject> objectsCopy = objects;
		entities.clear();
		for(Entry<String, RenderObject> e : objectsCopy.entrySet()) {
			processEntity(e.getValue());
		}
		Renderer.prepare();
		shader.start();
		shader.loadSkyColour(Renderer.RED, Renderer.GREEN, Renderer.BLUE);
		shader.loadLight(sun);
		shader.loadViewMatrix();
		Renderer.render(entities);
		shader.stop();
		entities.clear();
		Main.displayManager.updateDisplay();
	}
	
	public static void processEntity(RenderObject object) {
		TexturedModel entityModel = object.model;
		List<Entity> batch = new ArrayList<>(object.instances.values());
		entities.put(entityModel, batch);
	}
	
	public static void setObject(String parent, RenderObject object) {
		objects.put(parent, object);
	}
	
	public static void delObject(String parent) {
		objects.remove(parent);
	}
	
	public static Set<String> getKeys() {
		return objects.keySet();
	}
	
	public static Collection<RenderObject> getValues() {
		return objects.values();
	}
	
	public static boolean pushInstance(String parent, Entity entity) {
		if(objects.containsKey(parent)) {
			RenderObject ro = objects.get(parent);
			ro.instances.put(entity.getName(), entity);
			objects.put(parent, ro);
			return true;
		}
		return false;
	}
	
	public static boolean setInstance(String parent, Entity entity) {
		if(objects.containsKey(parent)) {
			RenderObject ro = objects.get(parent);
			ArrayList<String> toRemove = new ArrayList<String>();
			for(String name : ro.instances.keySet()) {
				if(name.equals(entity.getName())) {
					toRemove.add(name);
				}
			}
			for(String s : toRemove) {
				ro.instances.remove(s);
			}
			ro.instances.put(entity.getName(), entity);
			objects.put(parent, ro);
			return true;
		}
		return false;
	}
	
	public static boolean deleteInstance(String parent, String name) {
		if(objects.containsKey(parent)) {
			RenderObject ro = objects.get(parent);
			ArrayList<Entity> toRemove = new ArrayList<Entity>();
			for(Entity e : ro.instances.values()) {
				if(e.getName().equals(name)) {
					toRemove.add(e);
				}
			}
			for(Entity e : toRemove) {
				ro.instances.remove(e.getName());
			}
			objects.put(parent, ro);
			return true;
		}
		return false;
	}
	
	public static Entity getInstance(String parent, String name) {
		if(objects.containsKey(parent)) {
			RenderObject ro = objects.get(parent);
			for(Entity e : ro.instances.values()) {
				if(e.getName().equals(name)) {
					return ro.instances.get(e.getName());
				}
			}
		}
		return null;
	}
	
	public static void cleanUp() {
		shader.cleanUp();
	}

	private static final float speed = 0.05f;
	private static final float sensitivity = 0.1f;

	public static void move() {
		long window = Main.displayManager.window;
		if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_W) == GLFW.GLFW_PRESS) {
			camera.position.x -= Math.sin(Math.toRadians(camera.yaw)) * speed;
			camera.position.z += Math.cos(Math.toRadians(camera.yaw)) * speed;
		}
		if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_S) == GLFW.GLFW_PRESS) {
			camera.position.x += Math.sin(Math.toRadians(camera.yaw)) * speed;
			camera.position.z -= Math.cos(Math.toRadians(camera.yaw)) * speed;
		}
		if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_A) == GLFW.GLFW_PRESS) {
			camera.position.x -= Math.cos(Math.toRadians(camera.yaw)) * speed;
			camera.position.z -= Math.sin(Math.toRadians(camera.yaw)) * speed;
		}
		if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_D) == GLFW.GLFW_PRESS) {
			camera.position.x += Math.cos(Math.toRadians(camera.yaw)) * speed;
			camera.position.z += Math.sin(Math.toRadians(camera.yaw)) * speed;
		}
		if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_SPACE) == GLFW.GLFW_PRESS) {
			camera.position.y += speed;
		}
		if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_LEFT_SHIFT) == GLFW.GLFW_PRESS) {
			camera.position.y -= speed;
		}

		if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_UP) == GLFW.GLFW_PRESS) {
			camera.pitch -= sensitivity;
		}
		if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_DOWN) == GLFW.GLFW_PRESS) {
			camera.pitch += sensitivity;
		}
		if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_LEFT) == GLFW.GLFW_PRESS) {
			camera.yaw -= sensitivity;
		}
		if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_RIGHT) == GLFW.GLFW_PRESS) {
			camera.yaw += sensitivity;
		}
	}
	
}
