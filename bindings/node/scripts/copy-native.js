// Copies the cargo-built cdylib next to index.js under the platform's .node name.
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..', '..', '..', 'target', 'release');
const ext = { darwin: 'dylib', linux: 'so', win32: 'dll' }[process.platform];
const pre = process.platform === 'win32' ? '' : 'lib';
const src = path.join(root, `${pre}bodi_node.${ext}`);
const dst = path.join(__dirname, '..', `mahabodi.${process.platform}-${process.arch}.node`);
fs.copyFileSync(src, dst);
console.log(`copied ${src} -> ${dst}`);
