import 'package:tfx/src/support/common/sequential_processor.dart';
import 'package:tfx/src/support/tensorbuffer/tensor_buffer.dart';

/// https://github.com/tensorflow/tflite-support/blob/v0.4.1/tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/common/TensorProcessor.java

class TensorProcessor extends SequentialProcessor<TensorBuffer> {
  TensorProcessor(super.operatorList);
}
