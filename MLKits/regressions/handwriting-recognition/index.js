require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');

const mnist = require('mnist-data');
const mnistData = mnist.training(0, 1);
// console.log(mnistData);
// console.log(mnistData.images.values); //28 x 28 grid of pixels so we will make it one long array of 784
const features = mnistData.images.values.map(image => _.flatMap(image));
// console.log(features);
console.log(mnistData.labels.values); //gives the value of label in mnist dataset
