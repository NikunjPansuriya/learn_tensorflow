import React from "react";
import ReactDOM from "react-dom";

import "regenerator-runtime/runtime";

import Temperature from "./components/supervised/Temperature";

ReactDOM.render((
  <div>
    <h1>Hello Tensorflow</h1>
    <Temperature />
  </div>
), document.getElementById("app"));