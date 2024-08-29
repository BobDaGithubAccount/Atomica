package net.atomica.engine.entities;

import java.util.UUID;

public class Player {

    public float x;
    public float y;
    public float z;

    public float pitch;
    public float yaw;
    public float roll;

    public String name;
    public UUID connectionUUID;

    public Player(float x, float y, float z, float pitch, float yaw, float roll, String name, UUID connectionUUID) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.pitch = pitch;
        this.yaw = yaw;
        this.roll = roll;
        this.name = name;
        this.connectionUUID = connectionUUID;
    }
}
