const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
    constructor(features, labels, options) {
        this.features = tf.tensor(features);
        this.labels = tf.tensor(labels);

        this.features = tf
            .ones([this.features.shape[0], 1])
            .concat(this.features, 1);

        //If you provide options learningRate then it'll be used else default to 0.1 from object.assign
        this.options = Object.assign(
            { learningRate: 0.1, iterations: 1000 },
            options
        );

        this.weights = tf.zeros([2, 1]); //Creates a new tensor with values [[0], [0]]
    }

    //Calculate values of m and b, and use those update our guess
    gradientDescent() {
        const currentGuesses = this.features.matMul(this.weights);
        const differences = currentGuesses.sub(this.labels);

        const slopes = this.features
            .transpose()
            .matMul(differences)
            .div(this.features.shape[0]);

        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }

    //upto the train method to repeatedly call gradientDescent to get optimal solution of m and b
    train() {
        //don't run it for ever as you might be diverging in some cases
        for (let i = 0; i < this.options.iterations; i++) {
            this.gradientDescent();
        }
    }
}

module.exports = LinearRegression;
