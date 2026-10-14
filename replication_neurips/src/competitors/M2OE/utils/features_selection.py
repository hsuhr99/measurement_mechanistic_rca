import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

class FeaturesSelector(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        batch_n = x[y==0]
        batch_a = x[y==1]

        with tf.GradientTape() as tape:
            y_pred = self(batch_a, training=True)
            distance = tf.sqrt(tf.reduce_sum((batch_a - batch_n) ** 2, axis=1))
            subspace_distance = tf.sqrt(tf.reduce_sum(((batch_a - batch_n) ** 2) * y_pred, axis=1))
            dim_distance = tf.reduce_mean((distance / (subspace_distance + 1e-6)) - 1)
            penalty = - tf.math.log(1 - (tf.reduce_sum(y_pred) / batch_n.shape[1]) + 1e-6)
            loss = dim_distance + penalty

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}