const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
    constructor(features, labels, options) {
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.mseHistory = [];
        this.bHistory = [];

        // this.features = tf.tensor(features);
        // this.features = tf
        //     .ones([this.features.shape[0], 1])
        //     .concat(this.features, 1);

        //If you provide options learningRate then it'll be used else default to 0.1 from object.assign
        this.options = Object.assign(
            { learningRate: 0.1, iterations: 1000 },
            options
        );

        this.weights = tf.zeros([this.features.shape[1], 1]); //Creates a new tensor with values [[0], [0]]
    }

    //Calculate values of m and b, and use those update our guess
    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights);
        const differences = currentGuesses.sub(labels);

        const slopes = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0]);

        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }

    //upto the train method to repeatedly call gradientDescent to get optimal solution of m and b
    train() {
        const batchQuantity = Math.floor(
            this.features.shape[0] / this.options.batchSize
        );

        // console.log(this.options.learningRate);
        this.bHistory.push(this.weights.get(0, 0));

        //don't run it for ever as you might be diverging in some cases
        for (let i = 0; i < this.options.iterations; i++) {
            for (let j = 0; j < batchQuantity; j++) {
                const startIndex = j * this.options.batchSize;
                const { batchSize } = this.options;

                const featureSlice = this.features.slice(
                    [startIndex, 0],
                    [batchSize, -1]
                );
                const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

                this.gradientDescent(featureSlice, labelSlice);
            }

            this.recordMSE();
            this.updateLearningRate();
        }
    }

    predict(observations) {
        return this.processFeatures(observations).matMul(this.weights);
    }

    test(testFeatures, testLabels) {
        testFeatures = this.processFeatures(testFeatures);
        testLabels = tf.tensor(testLabels);

        const predictions = testFeatures.matMul(this.weights);
        // predictions.print();

        const res = testLabels.sub(predictions)
            .pow(2)
            .sum()
            .get()

        const tot = testLabels.sub(testLabels.mean())
            .pow(2)
            .sum()
            .get();

        return 1 - res / tot;
    }

    processFeatures(features) {
        features = tf.tensor(features);

        if (this.mean && this.variance) {
            features = features.sub(this.mean).div(this.variance.pow(0.5));
        } else {
            features = this.standardize(features);
        }
        features = tf.ones([features.shape[0], 1]).concat(features, 1);
        return features;
    }

    standardize(features) {
        const { mean, variance } = tf.moments(features, 0);

        this.mean = mean;
        this.variance = variance;

        return features.sub(mean).div(variance.pow(0.5));
    }

    recordMSE() {
        const mse = this.features
            .matMul(this.weights)
            .sub(this.labels)
            .pow(2)
            .sum()
            .div(this.features.shape[0])
            .get()

        this.mseHistory.unshift(mse); //same as push but at first place instead of last
    }

    updateLearningRate() {
        if (this.mseHistory.length < 2) {
            return;
        }

        if (this.mseHistory[0] > this.mseHistory[1]) {
            this.options.learningRate /= 2;
        } else {
            this.options.learningRate *= 1.05;
        }
    }
}

module.exports = LinearRegression;
