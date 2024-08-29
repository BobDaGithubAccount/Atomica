package net.atomica.engine.renderEngine;

import net.atomica.logging.Logger;
import org.lwjgl.BufferUtils;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL30;
import org.lwjgl.stb.STBImage;
import org.lwjgl.system.MemoryStack;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;

public class TextureLoader {

    public static int loadTexture(String fileName) {
        int width, height;
        ByteBuffer imageBuffer;

        String s = "net/atomica/engine/resources/" + fileName + "/" + fileName + ".png";
        Logger.info("Loading texture: " + s);

        try (InputStream in = TextureLoader.class.getClassLoader().getResourceAsStream(s)) {
            if (in == null) {
                throw new RuntimeException("Texture file not found: " + s);
            }

            imageBuffer = inputStreamToByteBuffer(in);
        } catch (Exception e) {
            Logger.severe("Failed to load texture: " + e.getMessage());
            e.printStackTrace();
            return -1;
        }

        ByteBuffer image;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer widthBuffer = stack.mallocInt(1);
            IntBuffer heightBuffer = stack.mallocInt(1);
            IntBuffer channelsBuffer = stack.mallocInt(1);

            image = STBImage.stbi_load_from_memory(imageBuffer, widthBuffer, heightBuffer, channelsBuffer, 4);
            if (image == null) {
                throw new RuntimeException("Failed to decode texture file: " + s + "\n" + STBImage.stbi_failure_reason());
            }

            width = widthBuffer.get();
            height = heightBuffer.get();
        }

        int textureID = GL11.glGenTextures();

        GL11.glBindTexture(GL11.GL_TEXTURE_2D, textureID);

        GL11.glTexImage2D(GL11.GL_TEXTURE_2D, 0, GL11.GL_RGBA, width, height, 0, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, image);

        GL30.glGenerateMipmap(GL11.GL_TEXTURE_2D);

        GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_MIN_FILTER, GL11.GL_LINEAR_MIPMAP_LINEAR);
        GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_MAG_FILTER, GL11.GL_LINEAR);
        GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_WRAP_S, GL11.GL_REPEAT);
        GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_WRAP_T, GL11.GL_REPEAT);

        GL11.glBindTexture(GL11.GL_TEXTURE_2D, 0);

        STBImage.stbi_image_free(image);

        return textureID;
    }

    private static ByteBuffer inputStreamToByteBuffer(InputStream inputStream) throws Exception {
        byte[] buffer = new byte[16384];
        int bytesRead;
        ByteBuffer byteBuffer = BufferUtils.createByteBuffer(8192);

        while ((bytesRead = inputStream.read(buffer, 0, buffer.length)) != -1) {
            if (byteBuffer.remaining() < bytesRead) {
                ByteBuffer newBuffer = BufferUtils.createByteBuffer(byteBuffer.capacity() * 2);
                byteBuffer.flip();
                newBuffer.put(byteBuffer);
                byteBuffer = newBuffer;
            }
            byteBuffer.put(buffer, 0, bytesRead);
        }

        byteBuffer.flip();
        return byteBuffer;
    }}