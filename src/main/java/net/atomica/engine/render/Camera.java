package net.atomica.engine.render;

import org.joml.Matrix4f;
import org.joml.Vector3f;

public class Camera {
    Vector3f position;
    private float pitch, yaw, roll;

    public Camera(Vector3f position, float pitch, float yaw, float roll) {
        this.position = position;
        this.pitch = pitch;
        this.yaw = yaw;
        this.roll = roll;
    }

    public Matrix4f getViewMatrix() {
        Matrix4f viewMatrix = new Matrix4f();
        viewMatrix.identity();
        viewMatrix.rotate((float) Math.toRadians(pitch), new Vector3f(1, 0, 0));
        viewMatrix.rotate((float) Math.toRadians(yaw), new Vector3f(0, 1, 0));
        viewMatrix.rotate((float) Math.toRadians(roll), new Vector3f(0, 0, 1));
        viewMatrix.translate(-position.x, -position.y, -position.z);
        return viewMatrix;
    }

    public void move(Vector3f direction) {
        position.add(direction);
    }

    public void rotate(float pitchDelta, float yawDelta, float rollDelta) {
        this.pitch += pitchDelta;
        this.yaw += yawDelta;
        this.roll += rollDelta;
    }
}
