const data = tf.tensor([1, 2, 3]);
const otherData = tf.tensor([4, 5, 6]);

data.shape;

//the output is a new tensor
data.add(otherData);
data.sub(otherData);
data.mul(otherData);
data.div(otherData);

//element wise operatiosn work on multi dimensional tensors
const data2 = tf.tensor([[1, 2, 3], [4, 5, 6]]);
const otherData2 = tf.tensor([[1, 2, 3], [4, 5, 6]]);
data2.add(otherData2);

//if shapes don't match we cannot do element wise operation
const noMatch = tf.tensor([4, 5, 6, 7]);
//data.add(noMatch) //won't work but in some cases we can

//example where shapes don't match but we still can
const data3 = tf.tensor([1, 2, 3]);
const otherData3 = tf.tensor([4]);
data3.add(otherData3); //This process is known as broadcasting

//Broadcasting works when - Take shape of both tensors & from right to left, the shapes are equal or one is '1'
//Smear one tensor over the other

//Broadcasting cannot work in below example
const data4 = tf.tensor([[1, 1, 1], [2, 2, 2]]);
const otherData4 = tf.tensor([1, 1]);
//data4.sub(otherData4);

//Print method to print out tensor
data.print();
////////////////////////////////////////////////////////////

//Element Access
const data = tf.tensor([10, 20, 30]);
data.get(0);

const data2 = tf.tensor([[10, 20, 30], [40, 50, 60]]);
data2.get(0, 1);

//Cannot set a value inside a tensor. There is no .set

//Slice Example
const data3 = tf.tensor([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90],
    [100, 110, 120],
]);
data3.slice([0, 1], [4, 2]);
data3.slice([0, 1], [data3.shape[0], 2]); //same as above as .shape gives [rows, columns]
data3.slice([0, 1], [-1, 2]); //-1 means as many rows as possible
const x = 1; //this is what we will vary most of the time - the column we want to get
data3.slice([0, x], [-1, 1]);

////////////////////////////////////////////////////////////
//Concat Example
const tensorA = tf.tensor([[1, 2, 3], [4, 5, 6]]);
const tensorB = tf.tensor([[7, 8, 9], [10, 11, 12]]);
tensorA.concat(tensorB);
tensorA.concat(tensorB, 1); //the number means axis of concatenation which by default is 0

////////////////////////////////////////////////////////////
//Play Jump Data Example
const jumpData = tf.tensor([
    [71, 72, 70],
    [70, 70, 70],
    [70, 70, 70],
    [70, 70, 70],
]);

const playerData = tf.tensor([[1, 160], [2, 160], [3, 160], [4, 160]]);

jumpData.sum(); //if we do this it blindly sums everything
jumpData.sum(1, 0); //to sum along an axis

//jumpData.sum(1).concat(playerData); //sum reduces dimension to 1
jumpData.sum(1, true).concat(playerData, 1);

jumpData.sum(1).expandDims(); //Increases dimension by 1
jumpData.sum(1).expandDims(1); //Expand along axis 1
jumpData
    .sum(1)
    .expandDims(1)
    .concat(playerData, 1); //1 so that they join on horizontal direction
////////////////////////////////////////////////////////////

//Standarization example - mean and variance (Sq. Rt variance -> Std. Deviation)
const numbers = tf.tensor([
    [1, 2],
    [3, 4],
    [5, 6]
]);

//moments works the same way as sum i.e. on all values so need to pass axis
const { mean, variance } = tf.moments(numbers, 0);
mean
variance

numbers.sub(mean).div(variance.pow(0.5));