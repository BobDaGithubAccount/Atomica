package net.atomica.logging;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.logging.FileHandler;
import java.util.logging.SimpleFormatter;

public class Logger {
    private static java.util.logging.Logger logger;

    public static void setLogger(java.util.logging.Logger logger1) {
        logger = logger1;
    }

    public static void info(String message) {
        if (logger != null) {
            logger.info(message);
        }
    }

    public static void warning(String message) {
        if (logger != null) {
            logger.warning(message);
        }
    }

    public static void severe(String message) {
        if (logger != null) {
            logger.severe(message);
        }
    }

    public static void info(String message, Throwable throwable) {
        if (logger != null) {
            logger.log(java.util.logging.Level.INFO, message, throwable);
        }
    }

    public static void warning(String message, Throwable throwable) {
        if (logger != null) {
            logger.log(java.util.logging.Level.WARNING, message, throwable);
        }
    }

    public static void severe(String message, Throwable throwable) {
        if (logger != null) {
            logger.log(java.util.logging.Level.SEVERE, message, throwable);
        }
    }

    public static java.util.logging.Logger constructLogger(String name) {
        try {
            java.util.logging.Logger logger = java.util.logging.Logger.getLogger(name);
            logger.setLevel(null);
            logger.setUseParentHandlers(true);

            File jarFile = new File(Logger.class.getProtectionDomain().getCodeSource().getLocation().getPath());
            File logsDir = new File(jarFile.getParent(), "logs");
            if (!logsDir.exists()) {
                logsDir.mkdirs();
            }

            String dateTime = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            File logFile = new File(logsDir, dateTime + ".txt");

            FileHandler fileHandler = new FileHandler(logFile.getPath(), true);
            fileHandler.setFormatter(new SimpleFormatter());
            logger.addHandler(fileHandler);

            return logger;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static void log(String s) {
        info(s);
    }

}