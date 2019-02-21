// disable in browser
// require('@tensorflow/tfjs-node');
// const tf = require('@tensorflow/tfjs');

class LogisticRegression {
    constructor(options) {
        this.options    = Object.assign({learningRate: 0.1, iterations: 1000, batchSize: 3}, options);
        this.mseHistory = [];
        this.weights    = null;
    }

    train(features, labels) {
        labels = tf.tensor(labels);

        features = this._processFeature(features);

        if (!this.weights) {
            this.weights = tf.zeros([features.shape[1], labels.shape[1]]);
        }


        const {batchSize, iterations} = this.options;
        const batQuantity             = Math.floor(features.shape[0] / batchSize);

        for (let i = 0; i < iterations; i++) {
            console.log(i);
            for (let j = 0; j < batQuantity; j++) {
                const featuresSliced = features.slice([j * batchSize], [batchSize, -1]);
                const labelsSliced   = labels.slice([j * batchSize], [batchSize, -1]);
                this._gradientDescent(featuresSliced, labelsSliced);
            }
            this._optimizeLearningRate()
        }
    }

    _gradientDescent(features, labels) {
        const mse    = features.matMul(this.weights).softmax().sub(labels);
        const slopes = features.transpose().matMul(mse);

        const diff = slopes.div(features.shape[0])
            .mul(2)
        ;

        this.weights = this.weights.sub(diff.mul(this.options.learningRate));
    }

    _standardize(features) {

        if (!this.mean) {
            const {mean, variance} = tf.moments(features, 0);

            const filler = variance
                .cast('bool')
                .logicalNot()
                .cast('float32');

            this.mean     = mean;
            this.variance = variance.add(filler);
        }

        return features.sub(this.mean).div(this.variance.pow(0.5));
    }

    test(testFeature, testLabel) {
        const labels      = tf.tensor(testLabel).argMax(1);
        const predictions = this.predict(testFeature);
        const incorrect   = predictions.notEqual(labels).sum().get();


        return (predictions.shape[0] - incorrect) / predictions.shape[0]
    }

    _optimizeLearningRate() {
        if (this.mseHistory.length < 2) {
            return
        }

        this.options.learningRate *= this.mseHistory[0] > this.mseHistory[1] ? 0.5 : 1.05;
    }

    _processFeature(features) {
        if (!features.shape) {
            features = tf.tensor(features);
        }


        // Standard values are available in the library (using normalization), No needs for this
        // features = this._standardize(features);

        features = tf.ones([features.shape[0], 1]).concat(features, 1);

        return features;
    }

    predict(values) {
        let a = this._processFeature(values);
        return a.matMul(this.weights).softmax().argMax(1);
    }
}

// disable in browser
// module.exports = LogisticRegression;
