
const CopyPlugin = require("copy-webpack-plugin");

module.exports = function override(config, env) {
  config.plugins.push(new CopyPlugin({
    // Use copy plugin to copy *.wasm to output folder.
    patterns: [{ from: 'node_modules/onnxruntime-web/dist/*.wasm', to: 'static/js/[name][ext]' }]
  }));
  return config;
};
