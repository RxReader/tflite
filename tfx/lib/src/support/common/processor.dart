/// https://github.com/tensorflow/tflite-support/blob/v0.4.1/tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/common/Processor.java

/// Processes T object with prepared [Operator].
abstract class Processor<T> {
  T process(T input);
}
