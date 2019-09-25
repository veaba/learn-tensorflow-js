console.log("hello tfjs")

// Create a buffer and set values at particular indices.
const buffer = tf.buffer([2, 2]);
buffer.set(3, 0, 0);
buffer.set(5, 1, 0);

// Convert the buffer back to a tensor.
buffer.toTensor().print();