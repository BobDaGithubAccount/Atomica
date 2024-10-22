bash .github/workflows/setup.sh
cd lib
wasm-pack build --target web
cd ..
rm -r dist
mkdir dist
cp src/index.html dist/index.html
cp -r lib/pkg/* dist/