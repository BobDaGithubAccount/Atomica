package net.atomica.engine.logic.events;

import net.atomica.Main;
import net.atomica.engine.logic.Event;
import net.atomica.engine.logic.EventHandler;
import net.atomica.engine.logic.EventType;
import net.atomica.engine.logic.KeyPressEvent;
import org.lwjgl.glfw.GLFW;

public class FullscreenToggleListener implements EventHandler {

    @Override
    public boolean run(Event event) {
        KeyPressEvent keyPressEvent = (KeyPressEvent) event;
        if ((int) keyPressEvent.getEventData()[0] == GLFW.GLFW_KEY_F11) {
            Main.RenderEngineContext.displayManager.toggleFullscreen();
        }
        return false;
    }

    @Override
    public EventType getEventType() {
        return EventType.KEY_PRESS;
    }
}
