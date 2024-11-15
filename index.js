import init, * as wasm_bindgen from './src.js';

init().then(() => {
    console.log("Wasm module initialized without standard control flow error?");
}).catch(err => {
    if (err.message.includes("Using exceptions for control flow")) {
        console.warn("Ignoring expected Wasm initialisation exception and proceeding.");
    } else {
        console.error("Error initializing wasm module:", err);
        throw err;
    }
}).finally(() => {

    console.log("Wasm module initialised!");

    // if (typeof window.resize_callback === 'undefined' && typeof wasm_bindgen.resize_callback !== 'undefined') {
    //     console.log("Setting resize callback");
    //     window.resize_callback = wasm_bindgen.resize_callback;
    // }
    // const resizableContainer = document.getElementById('resizable-container');
    // function resizeCanvas() {
    //     const canvas = document.getElementById('canvas');
    //     const width = resizableContainer.clientWidth;
    //     const height = resizableContainer.clientHeight;
    //     if (typeof window.resize_callback === 'function') {
    //         window.resize_callback(width, height);
    //     }
    //     canvas.width = width;
    //     canvas.height = height;
    // }

    // if (resizableContainer) {
    //     const resizeObserver = new ResizeObserver(() => {
    //         resizeCanvas();
    //     });

    //     resizeObserver.observe(resizableContainer);
    // }

    // window.addEventListener('load', resizeCanvas);

});