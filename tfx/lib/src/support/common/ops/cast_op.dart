import 'package:tfx/src/bindings/c/c_api_types.dart';
import 'package:tfx/src/support/common/tensor_operator.dart';
import 'package:tfx/src/support/tensorbuffer/tensor_buffer.dart';

/// https://github.com/tensorflow/tflite-support/blob/v0.4.1/tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/common/ops/CastOp.java

/// Casts a [TensorBuffer] to a specified data type.
class CastOp implements TensorOperator {
  /// Constructs a CastOp.
  /// Note: For only converting type for a certain [TensorBuffer] on-the-fly rather than in
  /// a processor, please directly use [TensorBuffer.createFrom(buffer: buffer, dataType: dataType)].
  ///
  /// When this Op is executed, if the original {@link TensorBuffer} is already in
  /// [destinationType], the original buffer will be directly returned.
  ///
  /// [destinationType] The type of the casted [TensorBuffer].
  const CastOp(int destinationType)
      : _destinationType = destinationType,
        assert(destinationType == TfLiteType.kTfLiteFloat32 || destinationType == TfLiteType.kTfLiteUInt8);

  final int /*TfLiteType*/ _destinationType;

  @override
  TensorBuffer apply(TensorBuffer input) {
    if (input.dataType == _destinationType) {
      return input;
    }
    return TensorBuffer.createFrom(input, _destinationType);
  }
}
