package net.atomica.engine.logic;

import java.util.Arrays;

public class KeyPressEvent implements Event {
    Object[] eventData = new Object[1];

    public KeyPressEvent(int key) {
        this.eventData[0] = key;
    }

    @Override
    public EventType getEventType() {
        return EventType.KEY_PRESS;
    }

    @Override
    public Object[] getEventData() {
        return eventData;
    }

    @Override
    public String toString() {
        return "KeyPressEvent{" +
                "eventData=" + Arrays.toString(eventData) +
                '}';
    }
}
