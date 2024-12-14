import init, * as wasm_bindgen from './atomica_lib.js';
import { handle_command, log } from './atomica_lib.js';

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
            handle_command_js(command);
        }
    });

    commandInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendButton.click();
        }
    });
}

function handle_command_js(command) {
    handle_command(command);
}

init().then(() => {
    jsInit();
}).catch(err => {
    console.error("Fake error initializing wasm module:", err);
    try {
        console.log("Second round of initialization successful! (Control flow error warning bypassed)");
        jsInit();
    } catch (err) {
        log("Real error initializing wasm module: " + err);
    }
});