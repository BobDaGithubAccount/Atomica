rustup update
rustup default stable
rustup target add wasm32-unknown-unknown
wasm-pack build --target web