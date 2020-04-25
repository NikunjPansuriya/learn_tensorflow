const path = require('path');
const HtmlWebpackPlugin = require("html-webpack-plugin");

module.exports = {

  mode: 'development',

  entry: './src/index.js',

  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bundle.js'
  },

  devServer: {
    contentBase: path.join(__dirname, 'dist'),
    compress: true,
    port: 8888
  },

  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader"
        }
      }, {
        test: /\.(png|jpe?g|gif)$/i,
        use: [
          {
            loader: 'file-loader',
          },
        ]
      }
    ]
  },

  resolve: {
    extensions: [".js", ".jsx"],
    alias: {
      Images: path.resolve(__dirname, "./src/images")
    }
  },

  plugins: [
    new HtmlWebpackPlugin({ template: './src/index.html' })
  ]
}