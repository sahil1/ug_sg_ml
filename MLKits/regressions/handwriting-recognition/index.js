require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');

const mnist = require('mnist-data');

function loadData() {
    const mnistData = mnist.training(0, 60000);
    // console.log(mnistData);
    // console.log(mnistData.images.values); //28 x 28 grid of pixels so we will make it one long array of 784
    const features = mnistData.images.values.map(image => _.flatMap(image));
    // console.log(features);

    // console.log(mnistData.labels.values); //gives the value of label in mnist dataset
    const encodedLabels = mnistData.labels.values.map(label => {
        const row = new Array(10).fill(0);
        row[label] = 1;
        return row;
    });
    // console.log(encodedLabels);

    return { features, labels: encodedLabels };
}

//For performance we don't need mnistData and so if we relieve reference on this it'll improve performance
const { features, labels } = loadData();

const regression = new LogisticRegression(features, labels, {
    learningRate: 1,
    iterations: 80,
    batchSize: 500,
});

regression.train();

const testMnistData = mnist.testing(0, 10000);
// console.log(testMnistData);
const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));
const testEncodedLabels = testMnistData.labels.values.map(label => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
});

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log('Accuracy is ', accuracy);

plot({
    x: regression.costHistory.reverse(),
});
