bash .github/workflows/setup.sh
pwd
cd lib
ls
wasm-pack build --target web
ls
cd ..
ls
rm -r dist
mkdir dist
cp src/index.html dist/index.html
cp -r lib/pkg/* dist/