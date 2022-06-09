import 'package:flutter/foundation.dart';
import 'package:tfx/src/support/tensorbuffer/tensor_buffer.dart';

/// https://github.com/tensorflow/tflite-support/blob/v0.4.1/tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/label/LabelUtil.java

/// Label operation utils.
class LabelUtil {
  const LabelUtil._();

  /// Maps an int value tensor to a list of string labels. It takes an array of strings as the
  /// dictionary. Example: if the given tensor is [3, 1, 0], and given labels is ["background",
  /// "apple", "banana", "cherry", "date"], the result will be ["date", "banana", "apple"].
  ///
  /// [tensorBuffer] A tensor with index values. The values should be non-negative integers, and
  ///   each value [x] will be converted to [labels[x + offset]]. If the tensor is
  ///   given as a float [TensorBuffer], values will be cast to integers. All values that are
  ///   out of bound will map to empty string.
  /// [labels] A list of strings, used as a dictionary to look up. The index of the array
  ///   element will be used as the key. To get better performance, use an object that implements
  ///   RandomAccess, such as [List].
  /// [offset] The offset value when look up int values in the [labels].
  /// return the mapped strings. The length of the list is [TensorBuffer.flatSize].
  static List<String> mapValueToLabels(TensorBuffer tensorBuffer, List<String> labels, int offset) {
    final List<int> values = tensorBuffer.getIntArray();
    if (kDebugMode) {
      print('values: $values');
    }
    final List<String> result = <String>[];
    for (int v in values) {
      final int index = v + offset;
      if (index < 0 || index >= labels.length) {
        result.add('');
      } else {
        result.add(labels[index]);
      }
    }
    return result;
  }
}
