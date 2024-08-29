package net.atomica.engine.renderEngine;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.*;

import net.atomica.engine.models.RawModel;
import net.atomica.engine.models.RawModels;
import net.atomica.engine.models.TexturedModel;
import net.atomica.engine.textures.ModelTexture;
import net.atomica.engine.entities.Entity;
import org.joml.Vector3f;
import org.lwjgl.BufferUtils;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL15;
import org.lwjgl.opengl.GL20;
import org.lwjgl.opengl.GL30;


public class Loader {

	public static Entity loadObj(String fileName, Vector3f location, Vector3f rotation, float scale) {
		RawModel model = GeometryLoader.loadObjModel(fileName);
		if(model==null) {
			String name = UUID.randomUUID().toString();
			RawModel model1 = RawModels.getErrorCube();
			ModelTexture texture = new ModelTexture(TextureLoader.loadTexture("essential"));
			TexturedModel texturedModel = new TexturedModel(model1, texture);
			Entity entity = new Entity(name, name, location, rotation, 1f);
			HashMap<String, Entity> instances = new HashMap<String, Entity>();
			instances.put(entity.getName(), entity);
			MasterRenderer.setObject(name, new RenderObject(instances, texturedModel));
			return entity;
		}
		ModelTexture texture = new ModelTexture(TextureLoader.loadTexture(fileName));
		TexturedModel texturedModel = new TexturedModel(model, texture);
		Entity entity = new Entity(fileName, fileName, location, rotation, 1f);
		HashMap<String, Entity> instances = new HashMap<String, Entity>();
		instances.put(entity.getName(), entity);
		MasterRenderer.setObject(fileName, new RenderObject(instances, texturedModel));
		return entity;
	}

	public static Entity loadRawModel(String fileName, RawModel model, Vector3f location, Vector3f rotation, float scale) {
		ModelTexture texture = new ModelTexture(TextureLoader.loadTexture(fileName));
		TexturedModel texturedModel = new TexturedModel(model, texture);
		Entity entity = new Entity(fileName, fileName, location, rotation, 1f);
		HashMap<String, Entity> instances = new HashMap<String, Entity>();
		instances.put(entity.getName(), entity);
		MasterRenderer.setObject(fileName, new RenderObject(instances, texturedModel));
		return entity;
	}
	
	private static List<Integer> vaos = new ArrayList<Integer>();
	private static List<Integer> vbos = new ArrayList<Integer>();
	private static List<Integer> textures = new ArrayList<Integer>();

	public static RawModel loadToVAO(float[] positions, int[] indices, float[] textureCoords, float[] normals) {
		int vaoID = createVAO();
		bindIndicesBuffer(indices);
		storeDataInAttributeList(0, 3, positions);
		storeDataInAttributeList(1, 2, textureCoords);
		storeDataInAttributeList(2, 3, normals);
		unbindVAO();
		return new RawModel(vaoID, indices.length, positions, indices, textureCoords, normals);
	}

	public static void cleanUp() {
		for(int vao : vaos) {
			GL30.glDeleteVertexArrays(vao);
		}
		for (int vbo : vbos) {
			GL15.glDeleteBuffers(vbo);
		}
		for(int texture : textures) {
			GL11.glDeleteTextures(texture);
		}
	}

	private static int createVAO() {
		int vaoID = GL30.glGenVertexArrays();
		vaos.add(vaoID);
		GL30.glBindVertexArray(vaoID);
		return vaoID;
	}

	private static void storeDataInAttributeList(int attributeNumber, int coordinateSize, float[] data) {
		int vboID = GL15.glGenBuffers();
		vbos.add(vboID);
		GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, vboID);
		FloatBuffer buffer = storeDataInFloatBuffer(data);
		GL15.glBufferData(GL15.GL_ARRAY_BUFFER, buffer, GL15.GL_STATIC_DRAW);
		GL20.glVertexAttribPointer(attributeNumber, coordinateSize, GL11.GL_FLOAT, false, 0, 0);
		GL15.glBindBuffer(GL15.GL_ARRAY_BUFFER, 0);
	}

	private static void unbindVAO() {
		GL30.glBindVertexArray(0);
	}
	
	private static void bindIndicesBuffer(int[] indices) {
		int vboID = GL15.glGenBuffers();
		vbos.add(vboID);
		GL15.glBindBuffer(GL15.GL_ELEMENT_ARRAY_BUFFER, vboID);
		IntBuffer buffer = storeDataInIntBuffer(indices);
		GL15.glBufferData(GL15.GL_ELEMENT_ARRAY_BUFFER, buffer, GL15.GL_STATIC_DRAW);
	}
	
	private static IntBuffer storeDataInIntBuffer(int[] data) {
		IntBuffer buffer = BufferUtils.createIntBuffer(data.length);
		buffer.put(data);
		buffer.flip();
		return buffer;
	}

	private static FloatBuffer storeDataInFloatBuffer(float[] data) {
		FloatBuffer buffer = BufferUtils.createFloatBuffer(data.length);
		buffer.put(data);
		buffer.flip();
		return buffer;
	}

}
