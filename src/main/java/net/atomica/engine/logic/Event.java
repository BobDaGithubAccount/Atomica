package net.atomica.engine.logic;

public interface Event {
    EventType getEventType();
    Object[] getEventData();
}
