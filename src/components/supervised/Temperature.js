import React, { useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

function Temperature() {

  useEffect(() => {
    visualizeData();
    startModelTraining();
  }, [])

  const generateData = () => {
    // Train data
    const data = Array(100).fill().map((x, i) => {
      return {
        celsius: i,
        fahrenheit: (i * 9 / 5) + 32
      }
    })

    return () => {
      return data;
    }
  }

  const getData = generateData();

  const visualizeData = () => {
    // prepare data for graph.
    const data = getData();
    const graphData = data.map(d => ({
      x: d.celsius,
      y: d.fahrenheit,
    }));

    // Plot chart
    tfvis.render.scatterplot(
      { name: 'Celsius v Fahrenheit' },
      { values: graphData },
      {
        xLabel: 'Celsius(°C)',
        yLabel: 'Fahrenheit(°F)',
        height: 300
      }
    );
  }

  const createModel = () => {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({ inputShape: [1], units: 1 }));

    // Add an output layer
    model.add(tf.layers.dense({ units: 1 }));

    return () => {
      return model;
    };
  }

  const getModel = createModel();

  const convertToTensor = (data) => {
    // Wrapping these calculations in a tidy will dispose any 
    // intermediate tensors.

    return tf.tidy(() => {
      // Step 1. Shuffle the data    
      tf.util.shuffle(data);

      const inputs = data.map(d => d.celsius)
      const labels = data.map(d => d.fahrenheit);

      // Step 2. Convert data to Tensor
      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

      console.log("***** input sensors *****");
      console.log(inputTensor);

      //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();

      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    });
  }

  const trainModel = async (model, inputs, labels) => {
    // Prepare the model for training.  
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });

    const batchSize = 32;
    const epochs = 50;

    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'],
        { height: 200, callbacks: ['onEpochEnd'] }
      )
    });
  }

  const startModelTraining = async () => {
    const data = getData();
    const tensorData = convertToTensor(data);
    console.log(tensorData);
    const { inputs, inputMax, inputMin, labels, labelMax, labelMin } = tensorData;

    const model = getModel();

    await trainModel(model, inputs, labels);
    console.log('Done Training');

    const [xs, preds] = tf.tidy(() => {

      const xs = tf.linspace(0, 1, 100);
      const preds = model.predict(xs.reshape([100, 1]));

      const unNormXs = xs
        .mul(inputMax.sub(inputMin))
        .add(inputMin);

      const unNormPreds = preds
        .mul(labelMax.sub(labelMin))
        .add(labelMin);

      // Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });


    const predictedPoints = Array.from(xs).map((val, i) => {
      return { x: val, y: preds[i] }
    });

    const originalPoints = data.map(d => ({
      x: d.celsius, y: d.fahrenheit,
    }));


    tfvis.render.scatterplot(
      { name: 'Celsius v Fahrenheit' },
      { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
      {
        xLabel: 'Celsius(°C)',
        yLabel: 'Fahrenheit(°F)',
        height: 300
      }
    );
  }

  return (
    <h2> Tensorflow is Fun </h2>
  )

}

export default Temperature;