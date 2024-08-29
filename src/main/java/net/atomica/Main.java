package net.atomica;

import net.atomica.engine.render.*;
import net.atomica.logging.Logger;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.stream.Stream;

public class Main {

    public static RenderEngine RenderEngineContext;
    public static File assetsDir;

    public static void main(String[] args) {
        Logger.setLogger(Logger.constructLogger("Atomica Engine"));
        Logger.log("Starting Atomica Engine...");
        copyAssets();
        RenderEngineContext = new RenderEngine();
        RenderEngineContext.run();
    }

    public static void copyAssets() {
        Path jarDir;
        try {
            jarDir = Paths.get(Main.class.getProtectionDomain().getCodeSource().getLocation().toURI()).getParent();
        } catch (URISyntaxException e) {
            e.printStackTrace();
            return;
        }

        File targetDir = new File(jarDir.toString() + "/assets");

        if (targetDir.exists()) {
            try (Stream<Path> paths = Files.walk(targetDir.toPath())) {
                paths.sorted(Comparator.reverseOrder())
                        .map(Path::toFile)
                        .forEach(File::delete);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        targetDir.mkdirs();
        Logger.log("Copying resources to " + targetDir);

        try (Stream<Path> paths = Files.walk(Paths.get(Main.class.getResource("/net/atomica/engine/resources").toURI()))) {
            Path resourceRoot = Paths.get(Main.class.getResource("/net/atomica/engine/resources").toURI());
            paths.forEach(sourcePath -> {
                Path relativeSourcePath = resourceRoot.relativize(sourcePath);
                Path targetPath = targetDir.toPath().resolve(relativeSourcePath.toString());
                try {
                    if (Files.isDirectory(sourcePath)) {
                        Files.createDirectories(targetPath);
                    } else {
                        try (InputStream in = Main.class.getResourceAsStream("/net/atomica/engine/resources/" + relativeSourcePath.toString())) {
                            Files.copy(in, targetPath, StandardCopyOption.REPLACE_EXISTING);
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
        assetsDir = targetDir;
    }

    public static void pollEvents() {

    }
}
