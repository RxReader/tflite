import 'dart:math' as math;

import 'package:tfx/src/bindings/c/c_api_types.dart';
import 'package:tfx/src/support/common/internal/support_preconditions.dart';
import 'package:tfx/src/support/tensorbuffer/tensor_buffer.dart';

/// https://github.com/tensorflow/tflite-support/blob/v0.4.1/tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/tensorbuffer/TensorBufferUint8.java

/// Represents data buffer with 8-bit unsigned integer values.
class TensorBufferUint8 extends TensorBuffer {
  TensorBufferUint8.dynamic() : super.dynamic();

  /// Creates a [TensorBufferUint8] with specified [shape].
  /// throw ArgumentError if [shape] has non-positive elements.
  TensorBufferUint8.shape(List<int> shape) : super.shape(shape);

  @override
  int get dataType => TfLiteType.kTfLiteUInt8;

  @override
  List<double> getFloatArray() {
    return getIntArray().map((int element) => element.toDouble()).toList();
  }

  @override
  double getFloatValue(int absIndex) {
    return getIntValue(absIndex).toDouble();
  }

  @override
  List<int> getIntArray() {
    final List<int> arr = List<int>.filled(flatSize, 0);
    for (int i = 0; i < flatSize; i++) {
      arr[i] = byteData.getUint8(i * typeSize) & 0xff;
    }
    return arr;
  }

  @override
  int getIntValue(int absIndex) {
    return byteData.getUint8(absIndex * typeSize) & 0xff;
  }

  @override
  void loadFloatArray(List<double> src, [List<int>? shape]) {
    shape ??= this.shape;
    SupportPreconditions.checkArgument(
      src.length == TensorBuffer.computeFlatSize(shape),
      'The size of the array to be loaded does not match the specified shape.',
    );
    resize(shape);

    for (int i = 0; i < src.length; i++) {
      byteData.setUint8(i * typeSize, math.max(math.min(src[i], 255.0), 0.0).floor() & 0xFF);
    }
  }

  @override
  void loadIntArray(List<int> src, [List<int>? shape]) {
    shape ??= this.shape;
    SupportPreconditions.checkArgument(
      src.length == TensorBuffer.computeFlatSize(shape),
      'The size of the array to be loaded does not match the specified shape.',
    );
    resize(shape);

    for (int i = 0; i < src.length; i++) {
      byteData.setUint8(i * typeSize, math.max(math.min(src[i], 255), 0) & 0xFF);
    }
  }
}
