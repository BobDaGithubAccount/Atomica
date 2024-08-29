package net.atomica;

import net.atomica.engine.renderEngine.DisplayManager;
import net.atomica.engine.renderEngine.MasterRenderer;
import net.atomica.engine.renderEngine.Renderer;
import net.atomica.logging.Logger;

import java.util.Timer;
import java.util.TimerTask;


public class Main {

    public static int fps = 0;
    public static DisplayManager displayManager;

    public void run() {
        Logger.setLogger(Logger.constructLogger("Atomica"));

        displayManager = new DisplayManager();
        displayManager.createDisplay();

        MasterRenderer.init();
        Renderer.initRenderer();

        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                displayManager.updateTitleWithFPS(fps);
                fps = 0;
            }
        }, 0, 1000);

        while (!displayManager.shouldClose()) {
            fps++;
            MasterRenderer.render();
            MasterRenderer.move();
        }

        MasterRenderer.cleanUp();
        displayManager.closeDisplay();
        System.exit(0);
    }

    public static void main(String[] args) throws Exception {
        new Main().run();
    }
}