package net.atomica.engine.logic;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Modifier;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

public class EventManager {

    public static Map<EventType, List<EventHandler>> eventListeners = new HashMap<>();

    public static void init() {
        String packageName = "net.atomica.engine.logic.events";
        try {
            List<Class<?>> classes = getClasses(packageName);
            for (Class<?> clazz : classes) {
                if (EventHandler.class.isAssignableFrom(clazz) && !Modifier.isAbstract(clazz.getModifiers())) {
                    EventHandler handler = (EventHandler) clazz.getDeclaredConstructor().newInstance();
                    registerEventHandler(handler);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void registerEventHandler(EventHandler handler) {
        EventType eventType = handler.getEventType();
        eventListeners.putIfAbsent(eventType, new ArrayList<>());
        eventListeners.get(eventType).add(handler);
    }

    private static List<Class<?>> getClasses(String packageName) throws URISyntaxException, ClassNotFoundException, IOException {
        List<Class<?>> classes = new ArrayList<>();
        String path = packageName.replace('.', '/');
        URL packageUrl = Thread.currentThread().getContextClassLoader().getResource(path);

        if (packageUrl == null) {
            throw new ClassNotFoundException("Package " + packageName + " not found");
        }

        if (packageUrl.getProtocol().equals("file")) {
            File directory = Paths.get(packageUrl.toURI()).toFile();
            if (directory.exists()) {
                File[] files = directory.listFiles();
                if (files != null) {
                    for (File file : files) {
                        if (file.getName().endsWith(".class")) {
                            String className = packageName + '.' + file.getName().substring(0, file.getName().length() - 6);
                            classes.add(Class.forName(className));
                        }
                    }
                }
            }
        } else if (packageUrl.getProtocol().equals("jar")) {
            String jarPath = packageUrl.getPath().substring(5, packageUrl.getPath().indexOf("!"));
            try (JarFile jarFile = new JarFile(jarPath)) {
                Enumeration<JarEntry> entries = jarFile.entries();
                while (entries.hasMoreElements()) {
                    JarEntry entry = entries.nextElement();
                    String entryName = entry.getName();
                    if (entryName.startsWith(path) && entryName.endsWith(".class")) {
                        String className = entryName.replace('/', '.').substring(0, entryName.length() - 6);
                        classes.add(Class.forName(className));
                    }
                }
            }
        }
        return classes;
    }

    public static void fireEvent(Event event) {
        List<EventHandler> handlers = eventListeners.get(event.getEventType());
        if (handlers != null) {
            for (EventHandler handler : handlers) {
                if (handler.run(event)) {
                    break;
                }
            }
        }
    }
}
