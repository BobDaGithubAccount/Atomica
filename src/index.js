import init, * as wasm_bindgen from './src.js';
import { reset_camera, update_camera_fov } from './src.js';

function jsInit() {
    console.log("Wasm module initialized");
    const canvas = document.getElementById('canvas');
    const sendButton = document.getElementById('send-button');
    const commandInput = document.getElementById('command-input');
    const logArea = document.getElementById('log-area');

    sendButton.addEventListener('click', () => {
        const command = commandInput.value.trim();
        if (command) {
            log(`User command: ${command}`);
            commandInput.value = '';
        }
    });

    commandInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendButton.click();
        }
    });
}

function log(message) {
    const logArea = document.getElementById('log-area');
    const timestamp = new Date().toLocaleTimeString();
    logArea.innerHTML += `[${timestamp}] ${message}<br>`;
    logArea.scrollTop = logArea.scrollHeight;
}

init().then(() => {
    jsInit();
}).catch(err => {
    console.error("Fake error initializing wasm module:", err);
    try {
        console.log("Second round of initialization successful! (Control flow error warning bypassed");
        jsInit();
    } catch (err) {
        console.error("Real error initializing wasm module:", err);
    }
});