const outputs = [
    [10, 0.5, 16, 1],
    [200, 0.5, 16, 4],
    [350, 0.5, 16, 4],
    [600, 0.5, 16, 5],
];

const predictionPoint = 300;
const k = 3;

function distance(point) {
    return Math.abs(point - predictionPoint);
}

_.chain(outputs)
    .map(row => [distance(row[0]), row[3]])
    .sortBy(row => row[0])
    .slice(0, k)
    .countBy(row => row[1])
    .toPairs()
    .sortBy(row => row[1])
    .last()
    .first()
    .parseInt()
    .value();

//Multi-Dimensional knn
const pointA = [1, 1];
const pointB = [4, 5];

_.chain(pointA)
    .zip(pointB)
    .map(([a, b]) => (a - b) ** 2)
    .sum()
    .value() ** 0.5;

//Initial and Last
const point = [350, 0.5, 16, 4];

_.initial(point);
_.last(point);

//Normalization
const points = [200, 150, 650, 430];

const min = _.min(points);
const max = _.max(points);

_.map(points, point => {
    return (point - min) / (max - min);
});
