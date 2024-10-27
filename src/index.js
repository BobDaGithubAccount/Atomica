import init, * as wasm_bindgen from './src.js';

function jsInit() {
    console.log("Wasm module initialized");

    if (typeof window.resize_callback === 'undefined' && typeof wasm_bindgen.resize_callback !== 'undefined') {
        console.log("Setting resize callback");
        window.resize_callback = wasm_bindgen.resize_callback;
    }

    const canvas = document.getElementById('canvas');

    function resizeCanvas() {
        const canvas = document.getElementById('canvas');
        const width = window.innerWidth;
        const height = window.innerHeight;
        canvas.width = width;
        canvas.height = height;

        if (typeof window.resize_callback === 'function') {
            window.resize_callback(width, height);
        }
    }

    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('load', resizeCanvas);
}
init().then(() => {
    jsInit();
}).catch(err => {
    if (err.message.includes("Using exceptions for control flow")) {
        console.warn("Ignoring expected Wasm initialization exception and proceeding.");
        try {
            jsInit();
        } catch (err) {
            console.error("Error in continuing initialisation:", err);
            throw err;
        }
    } else {
        console.error("Error initializing wasm module:", err);
        throw err;
    }
});
console.log("Wasm module initialization call made");
