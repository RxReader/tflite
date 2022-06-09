import 'dart:typed_data';

import 'package:tfx/src/bindings/c/c_api_types.dart';
import 'package:tfx/src/support/common/internal/support_preconditions.dart';
import 'package:tfx/src/support/tensorbuffer/tensor_buffer.dart';

/// https://github.com/tensorflow/tflite-support/blob/v0.4.1/tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/tensorbuffer/TensorBufferFloat.java

/// Represents data buffer with float values.
class TensorBufferFloat extends TensorBuffer {
  TensorBufferFloat.dynamic() : super.dynamic();

  /// Creates a [TensorBufferFloat] with specified [shape].
  ///
  /// throw ArgumentError if {@code shape} has non-positive elements.
  TensorBufferFloat.shape(List<int> shape) : super.shape(shape);

  @override
  int get dataType => TfLiteType.kTfLiteFloat32;

  @override
  List<double> getFloatArray() {
    final List<double> arr = List<double>.filled(flatSize, 0);
    for (int i = 0; i < flatSize; i++) {
      arr[i] = byteData.getFloat32(i * typeSize, Endian.little);
    }
    return arr;
  }

  @override
  double getFloatValue(int absIndex) {
    return byteData.getFloat32(absIndex * typeSize, Endian.little);
  }

  @override
  List<int> getIntArray() {
    return getFloatArray().map((double element) => element.floor()).toList();
  }

  @override
  int getIntValue(int absIndex) {
    return getFloatValue(absIndex).floor();
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
      byteData.setFloat32(i * typeSize, src[i], Endian.little);
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
      byteData.setFloat32(i * typeSize, src[i].toDouble(), Endian.little);
    }
  }
}
