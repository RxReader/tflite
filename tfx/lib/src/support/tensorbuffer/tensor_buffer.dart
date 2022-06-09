import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:tfx/src/bindings/c/c_api_types.dart';
import 'package:tfx/src/support/common/internal/support_preconditions.dart';
import 'package:tfx/src/support/tensorbuffer/tensor_buffer_float.dart';
import 'package:tfx/src/support/tensorbuffer/tensor_buffer_uint8.dart';

/// https://github.com/tensorflow/tflite-support/blob/v0.4.1/tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/tensorbuffer/TensorBuffer.java

/// Represents the data buffer for either a model's input or its output.
abstract class TensorBuffer {
  /// Constructs a fixed size [TensorBuffer] with specified [shape].
  /// throw ArgumentError if [shape] has non-positive elements.
  @protected
  TensorBuffer(List<int> shape) {
    _isDynamic = false;
    _allocateMemory(shape);
  }

  /// Creates an empty dynamic [TensorBuffer] with specified [TfLiteType]. The shape of the
  /// created [TensorBuffer] is {0}.
  ///
  /// Dynamic TensorBuffers will reallocate memory when loading arrays or data buffers of
  /// different buffer sizes. Here are some examples:
  ///
  /// // Creating a float dynamic TensorBuffer:
  /// TensorBuffer tensorBuffer = TensorBuffer.createDynamic(TfLiteType.kTfLiteFloat32);
  /// // Loading a float array:
  /// List<double> arr1 = <double>[1, 2, 3];
  /// tensorBuffer.loadFloatArray(arr1, <int>[arr1.length]);
  /// // loading another float array:
  /// List<double> arr2 = <double>[1, 2, 3, 4, 5];
  /// tensorBuffer.loadFloatArray(arr2, <int>[arr2.length]);
  /// // loading a third float array with the same size as arr2, assuming shape doesn't change:
  /// List<double> arr3 = <double>[5, 4, 3, 2, 1];
  /// tensorBuffer.loadFloatArray(arr3);
  /// // loading a forth float array with different size as arr3 and omitting the shape will result
  /// // in error:
  /// List<double> arr4 = <double>[3, 2, 1];
  /// tensorBuffer.loadFloatArray(arr4); // Error: The size of byte buffer and the shape do not match.
  ///
  /// [dataType] The dataType of the [TensorBuffer] to be created.
  factory TensorBuffer.createDynamic({
    required int /*TfLiteType*/ dataType,
  }) {
    switch (dataType) {
      case TfLiteType.kTfLiteFloat32:
        return TensorBufferFloat.dynamic();
      case TfLiteType.kTfLiteUInt8:
        return TensorBufferUint8.dynamic();
      default:
        throw ArgumentError('TensorBuffer does not support data type: $dataType');
    }
  }

  /// Creates a [TensorBuffer] with specified [shape] and [TfLiteType]. Here are some
  /// examples:
  ///
  /// // Creating a float TensorBuffer with shape {2, 3}:
  /// List<int> shape = <int>[2, 3];
  /// TensorBuffer tensorBuffer = TensorBuffer.createFixedSize(shape, TfLiteType.kTfLiteFloat32);
  ///
  /// // Creating an uint8 TensorBuffer of a scalar:
  /// List<int> shape = <int>[];
  /// TensorBuffer tensorBuffer = TensorBuffer.createFixedSize(shape, TfLiteType.kTfLiteUInt8);
  ///
  /// // Creating an empty uint8 TensorBuffer:
  /// List<int> shape = <int>[0];
  /// TensorBuffer tensorBuffer = TensorBuffer.createFixedSize(shape, TfLiteType.kTfLiteUInt8);
  ///
  /// The size of a fixed-size TensorBuffer cannot be changed once it is created.
  ///
  /// [shape] The shape of the [TensorBuffer] to be created.
  /// [dataType] The dataType of the [TensorBuffer] to be created.
  /// throw ArgumentError if [shape] has non-positive elements.
  factory TensorBuffer.createFixedSize({
    required List<int> shape,
    required int /*TfLiteType*/ dataType,
  }) {
    switch (dataType) {
      case TfLiteType.kTfLiteFloat32:
        return TensorBufferFloat(shape);
      case TfLiteType.kTfLiteUInt8:
        return TensorBufferUint8(shape);
      default:
        throw ArgumentError('TensorBuffer does not support data type: $dataType');
    }
  }

  /// Creates a [TensorBuffer] deep-copying data from another, with specified [TfLiteType].
  ///
  /// [buffer] the source [TensorBuffer] to copy from.
  /// [dataType] the expected [TfLiteType] of newly created [TensorBuffer].
  factory TensorBuffer.createFrom({
    required TensorBuffer buffer,
    required int /*TfLiteType*/ dataType,
  }) {
    TensorBuffer result;
    if (buffer.isDynamic) {
      result = TensorBuffer.createDynamic(dataType: dataType);
    } else {
      result = TensorBuffer.createFixedSize(shape: buffer.shape, dataType: dataType);
    }
    if (buffer.dataType == TfLiteType.kTfLiteFloat32 && dataType == TfLiteType.kTfLiteFloat32) {
      final List<double> data = buffer.getFloatArray();
      result.loadFloatArray(data, buffer.shape);
    } else {
      final List<int> data = buffer.getIntArray();
      result.loadIntArray(data, buffer.shape);
    }
    return result;
  }

  /// Constructs a dynamic [TensorBuffer] which can be resized.
  @protected
  TensorBuffer.dynamic() {
    _isDynamic = true;
    _allocateMemory(<int>[0]);
  }

  /// Where the data is stored.
  late ByteData _byteData;

  /// Shape of the tensor stored in this buffer.
  late List<int> _shape;

  /// Number of elements in the buffer. It will be changed to a proper value in the constructor.
  late int _flatSize = -1;

  /// Indicator of whether this buffer is dynamic or fixed-size. Fixed-size buffers will have
  /// pre-allocated memory and fixed size. While the size of dynamic buffers can be changed.
  late bool _isDynamic;

  /// Returns the data type of this buffer.
  int /*TfLiteType*/ get dataType;

  /// Returns the number of bytes of a single element in the array. For example, a float buffer will
  /// return 4, and a byte buffer will return 1.
  int get typeSize {
    switch (dataType) {
      case TfLiteType.kTfLiteFloat32:
      case TfLiteType.kTfLiteInt32:
        return 4;
      case TfLiteType.kTfLiteInt16:
        return 2;
      case TfLiteType.kTfLiteInt8:
      case TfLiteType.kTfLiteUInt8:
        return 1;
      case TfLiteType.kTfLiteInt64:
        return 8;
      case TfLiteType.kTfLiteBool:
        return -1;
      case TfLiteType.kTfLiteString:
        return -1;
    }
    throw ArgumentError.value(dataType, null, 'TfLiteType error: TfLiteType $dataType is not supported yet');
  }

  @protected
  ByteData get byteData => _byteData;

  /// Returns the data buffer.
  ByteBuffer get buffer => _byteData.buffer;

  /// Gets the current shape. (returning a copy here to avoid unexpected modification.)
  List<int> get shape {
    _assertShapeIsCorrect();
    return List<int>.of(_shape);
  }

  /// Gets the flatSize of the buffer.
  int get flatSize {
    _assertShapeIsCorrect();
    return _flatSize;
  }

  /// Returns if the {@link TensorBuffer} is dynamic sized (could resize arbitrarily).
  bool get isDynamic => _isDynamic;

  /// Allocates buffer with corresponding size of the [shape]. If shape is an empty array, this
  /// [TensorBuffer] will be created as a scalar and its flatSize will be 1.
  ///
  /// throw ArgumentError if [shape] has negative elements.
  void _allocateMemory(List<int> shape) {
    SupportPreconditions.checkArgument(isShapeValid(shape), 'Values in TensorBuffer shape should be non-negative.');

    // Check if the new shape is the same as current shape.
    final int newFlatSize = computeFlatSize(shape);
    _shape = List<int>.of(shape);
    if (_flatSize == newFlatSize) {
      return;
    }

    // Update to the new shape.
    _flatSize = newFlatSize;
    _byteData = ByteData(_flatSize * typeSize);
  }

  /// Verifies if the shape of the [TensorBuffer] matched the size of the underlying [ByteBuffer]
  void _assertShapeIsCorrect() {
    final int flatSize = computeFlatSize(_shape);
    SupportPreconditions.checkState(
        _byteData.lengthInBytes == typeSize * flatSize, 'The size of underlying ByteBuffer (${_byteData.lengthInBytes}) and the shape ($_shape) do not match. The ByteBuffer may have been changed.');
  }

  /// Returns a float array of the values stored in this buffer. If the buffer is of different types
  /// than float, the values will be converted into float. For example, values in [TensorBufferUint8]
  /// will be converted from uint8 to float.
  List<double> getFloatArray();

  /// Returns a float value at a given index. If the buffer is of different types than float, the
  /// value will be converted into float. For example, when reading a value from [TensorBufferUint8],
  /// the value will be first read out as uint8, and then will be converted from uint8 to float.
  ///
  /// For example, a TensorBuffer with shape [2, 3] that represents the following array,
  /// [[0.0f, 1.0f, 2.0f], [3.0f, 4.0f, 5.0f]].
  ///
  /// The fourth element (whose value is 3.0f) in the TensorBuffer can be retrieved by:
  /// double v = tensorBuffer.getFloatValue(3);
  ///
  /// [absIndex] The absolute index of the value to be read.
  double getFloatValue(int absIndex);

  /// Returns an int array of the values stored in this buffer. If the buffer is of different type
  /// than int, the values will be converted into int, and loss of precision may apply. For example,
  /// getting an int array from a [TensorBufferFloat] with values [400.32f, 23.04f], the output
  /// is [400, 23].
  List<int> getIntArray();

  /// Returns an int value at a given index. If the buffer is of different types than int, the value
  /// will be converted into int. For example, when reading a value from [TensorBufferFloat],
  /// the value will be first read out as float, and then will be converted from float to int. Loss
  /// of precision may apply.
  ///
  /// For example, a TensorBuffer with shape [2, 3] that represents the following array,
  /// [[0.0f, 1.0f, 2.0f], [3.0f, 4.0f, 5.0f]].
  ///
  /// The fourth element (whose value is 3.0f) in the TensorBuffer can be retrieved by:
  /// int v = tensorBuffer.getIntValue(3);
  /// Note that v is converted from 3.0f to 3 as a result of type conversion.
  ///
  /// [absIndex] The absolute index of the value to be read.
  int getIntValue(int absIndex);

  /// Loads an int array into this buffer with specific shape. If the buffer is of different types
  /// than int, the values will be converted into the buffer's type before being loaded into the
  /// buffer, and loss of precision may apply. For example, loading an int array with values [400,
  /// -23] into a [TensorBufferUint8] , the values will be clamped to [0, 255] and then be
  /// casted to uint8 by [255, 0].
  ///
  /// [src] The source array to be loaded.
  /// [shape] Shape of the tensor that [src] represents. If [shape] is null then [TensorBuffer.shape] is used.
  /// throw ArgumentError if the size of the array to be loaded does not match the specified shape.
  void loadIntArray(List<int> src, [List<int>? shape]);

  /// Loads a float array into this buffer with specific shape. If the buffer is of different types
  /// than float, the values will be converted into the buffer's type before being loaded into the
  /// buffer, and loss of precision may apply. For example, loading a float array into a
  /// [TensorBufferUint8] with values [400.32f, -23.04f], the values will be clamped to [0, 255] and
  /// then be casted to uint8 by [255, 0].
  ///
  /// [src] The source array to be loaded.
  /// [shape] Shape of the tensor that [src] represents. If [shape] is null then [TensorBuffer.shape] is used.
  /// throw ArgumentError if the size of the array to be loaded does not match the specified shape.
  void loadFloatArray(List<double> src, [List<int>? shape]);

  /// Loads a byte buffer into this [TensorBuffer] with specific shape.
  ///
  /// Important: The loaded buffer is a reference. DO NOT MODIFY. We don't create a copy here for
  /// performance concern, but if modification is necessary, please make a copy.
  ///
  /// For the best performance, always load a direct [ByteBuffer] or a [ByteBuffer]
  /// backed by an array.
  ///
  /// [buffer] The byte buffer to load.
  /// throw ArgumentError if the size of [buffer] and [typeSize] do not
  ///   match or the size of [buffer] and [flatSize] do not match.
  void loadBuffer(ByteBuffer buffer, [List<int>? shape]) {
    shape ??= _shape;
    SupportPreconditions.checkArgument(isShapeValid(shape), 'Values in TensorBuffer shape should be non-negative.');

    final int flatSize = computeFlatSize(shape);

    SupportPreconditions.checkArgument(
      ByteData.view(buffer).lengthInBytes == typeSize * flatSize,
      'The size of byte buffer and the shape do not match. buffer: ${ByteData.view(buffer).lengthInBytes} shape: ${typeSize * flatSize}',
    );

    if (!_isDynamic) {
      // Make sure the new shape fits the buffer size when TensorBuffer has fixed size.
      SupportPreconditions.checkArgument(listEquals<int>(shape, _shape));
    }

    // Update to the new shape, since shape dim values might change.
    _shape = List<int>.of(shape);
    _flatSize = flatSize;
    _byteData = ByteData.view(buffer);
  }

  /// For dynamic buffer, resize the memory if needed. For fixed-size buffer, check if the
  /// [shape] of src fits the buffer size.
  @protected
  void resize(List<int> shape) {
    if (_isDynamic) {
      _allocateMemory(shape);
    } else {
      // Make sure the new shape fits the buffer size when TensorBuffer has fixed size.
      SupportPreconditions.checkArgument(listEquals(shape, _shape));
      _shape = List<int>.of(shape);
    }
  }

  static bool isShapeValid(List<int> shape) {
    if (shape.isEmpty) {
      // This shape refers to a scalar.
      return true;
    }
    // This shape refers to a multidimensional array.
    for (int s in shape) {
      // All elements in shape should be non-negative.
      if (s < 0) {
        return false;
      }
    }
    return true;
  }

  /// Calculates number of elements in the buffer.
  static int computeFlatSize(List<int> shape) {
    int prod = 1;
    for (int s in shape) {
      prod = prod * s;
    }
    return prod;
  }
}
