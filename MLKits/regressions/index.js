require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg'],
});

// console.log(features, labels);

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 3,
    batchSize: 10
});

// regression.features.print();
regression.train();
const r2 = regression.test(testFeatures, testLabels);
plot({
    x: regression.mseHistory.reverse(),
    xLabel: 'Iteration #',
    yLabel: 'Mean Squared Error'
});

// plot({
//     x: regression.bHistory,
//     y: regression.mseHistory.reverse(),
//     xLabel: 'Value of b',
//     yLabel: 'Mean Squared Error'
// });

console.log('MSE History :', regression.mseHistory);
console.log('r2 is :', r2);
console.log(
    'Updated M is:',
    regression.weights.get(1, 0),
    ' Updated B is:',
    regression.weights.get(0, 0)
);

regression.predict([
    [120, 2, 380],
    [135, 2.1, 400]
]).print();