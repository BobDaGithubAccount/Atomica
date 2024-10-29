if ! command -v rustup &> /dev/null
then
    echo "rustup not found, installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
else
    echo "rustup is already installed"
fi

rustup update
rustup default stable
rustup target add wasm32-unknown-unknown
cargo install wasm-pack