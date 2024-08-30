package net.atomica.engine.logic.events;

import net.atomica.Main;
import net.atomica.engine.logic.Event;
import net.atomica.engine.logic.EventHandler;
import net.atomica.engine.logic.EventType;
import net.atomica.engine.logic.KeyPressEvent;
import org.lwjgl.glfw.GLFW;

public class WASDListener implements EventHandler {

    @Override
    public boolean run(Event event) {
        KeyPressEvent keyPressEvent = (KeyPressEvent) event;
        switch((int)keyPressEvent.getEventData()[0]) {
            case GLFW.GLFW_KEY_W:
                Main.RenderEngineContext.camera.position.z += 0.1f;
                break;
            case GLFW.GLFW_KEY_A:
                Main.RenderEngineContext.camera.position.x += 0.1f;
                break;
            case GLFW.GLFW_KEY_S:
                Main.RenderEngineContext.camera.position.z -= 0.1f;
                break;
            case GLFW.GLFW_KEY_D:
                Main.RenderEngineContext.camera.position.x -= 0.1f;
                break;
            case GLFW.GLFW_KEY_SPACE:
                Main.RenderEngineContext.camera.position.y += 0.1f;
                break;
            case GLFW.GLFW_KEY_LEFT_SHIFT:
                Main.RenderEngineContext.camera.position.y -= 0.1f;
                break;
        }
        return false;
    }

    @Override
    public EventType getEventType() {
        return EventType.KEY_PRESS;
    }
}
