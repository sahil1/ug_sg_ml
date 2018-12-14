const _ = require('lodash');

const loadData = () => {
    const randoms = _.range(0, 999999); //range creates array from 0 ... 999998

    return randoms; //if you comment this then memory usage will go down significantly as no significant allocation
    //js garbage collector behind this
};

const data = loadData();

debugger;
