const mnist              = require('mnist-data');
const fs                 = require('fs');
const LogisticRegression = require('./LogisticRegression');

const training_data     = mnist.training(0, 8000);
const test_data         = mnist.testing(0, 10000);
Array.prototype.flatMap = function (lambda) {

    return Array.prototype.concat.apply([], this.map(lambda));
};

const logisticRegression = new LogisticRegression({iterations: 100, batchSize: 8000});
logisticRegression.train(
    training_data.images.values.map(value => value.flatMap(i => i)),
    training_data.labels.values.map((val) => {
        const arr              = new Array(10).fill(0);
        arr[parseInt(val) - 1] = 1;
        return arr
    }),
);

logisticRegression.weights.array().then(data => {
    fs.writeFile('model.json', JSON.stringify(data));
    console.log('done');
});


console.log('correct % : ', logisticRegression.test(test_data.images.values.map(value => value.flatMap(i => i)),
    test_data.labels.values.map((val) => {
        const arr              = new Array(10).fill(0);
        arr[parseInt(val) - 1] = 1;
        return arr
    })));
