import 'package:tfx/src/support/label/tensor_label.dart';
import 'package:tfx/src/support/tensorbuffer/tensor_buffer.dart';

/// https://github.com/tensorflow/tflite-support/blob/v0.4.1/tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/label/ops/LabelAxisOp.java

/// Labels TensorBuffer with axisLabels for outputs.
///
/// Apply on a [TensorBuffer] to get a [TensorLabel] that could output a Map, which is
/// a pair of the label name and the corresponding TensorBuffer value.
class LabelAxisOp {
  const LabelAxisOp(this.axisLabels);

  /// Axis and its corresponding label names.
  final Map<int, List<String>> axisLabels;

  TensorLabel apply(TensorBuffer buffer) {
    return TensorLabel.fromMap(axisLabels: axisLabels, tensorBuffer: buffer);
  }
}
