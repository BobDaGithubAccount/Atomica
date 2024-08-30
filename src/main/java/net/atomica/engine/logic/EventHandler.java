package net.atomica.engine.logic;

public interface EventHandler {
    boolean run(Event event);
    EventType getEventType();
}
